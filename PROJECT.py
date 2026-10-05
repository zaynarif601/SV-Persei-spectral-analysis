#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#RV code


# In[1]:


"""
=============================================================================
  Spectral Synthesis Pipeline — SVPer
  - Convolve Kurucz template to R=80,000
  - Continuum normalisation
  - Cross-correlation → Gaussian CCF → RV + uncertainty
  - CCF plot in km/s (±500) and Angstroms
  - Observed vs synthetic comparison + residuals
=============================================================================
"""
get_ipython().run_line_magic('matplotlib', 'qt')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymoog
import astropy.units as u
from astropy.constants import c as c_light
from astropy.coordinates import SpectralCoord
from astropy.nddata import StdDevUncertainty
from astropy.convolution import Gaussian1DKernel, convolve
from specutils import Spectrum1D
from specutils.analysis import template_correlate
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit


# =============================================================================
# HELPER 1 — Convolve synthetic to R=80,000
# =============================================================================

def convolve_to_resolution(synth_wave, synth_flux, resolution=80000):
    """
    Blur synthetic spectrum to match instrumental resolving power R.
    FWHM = lambda_mid / R  →  sigma = FWHM / 2.355
    """
    mean_wave    = np.mean(synth_wave)
    fwhm_wave    = mean_wave / resolution
    sigma_wave   = fwhm_wave / 2.355
    step_size    = np.mean(np.diff(synth_wave))
    sigma_pixels = sigma_wave / step_size

    print(f"  Convolution : R={resolution}  "
          f"FWHM={fwhm_wave:.4f} Å  sigma={sigma_pixels:.3f} px")

    kernel = Gaussian1DKernel(stddev=sigma_pixels)
    return convolve(synth_flux, kernel, boundary='extend')


# =============================================================================
# HELPER 2 — Continuum normalisation
# =============================================================================

def local_normalise(wave, flux, n_knots=6):
    """
    Fit a cubic spline to the 90th-percentile flux in each wavelength chunk
    and divide the spectrum by it. More robust than a single global percentile
    for Cepheid supergiants with many strong absorption lines.
    """
    n_chunks   = n_knots + 1
    chunk_size = len(wave) // n_chunks
    knot_waves, knot_fluxes = [], []

    for k in range(n_chunks):
        lo         = k * chunk_size
        hi         = min(lo + chunk_size, len(wave))
        chunk_flux = flux[lo:hi]
        chunk_wave = wave[lo:hi]
        p90_idx    = np.argmin(
            np.abs(chunk_flux - np.percentile(chunk_flux, 90))
        )
        knot_waves.append(chunk_wave[p90_idx])
        knot_fluxes.append(chunk_flux[p90_idx])

    spl       = UnivariateSpline(knot_waves, knot_fluxes, k=3, s=0, ext=3)
    continuum = np.maximum(spl(wave), 1e-10)
    return flux / continuum


# =============================================================================
# HELPER 3 — CCF plots in km/s and Angstroms
# =============================================================================

def plot_ccf(lags_kms, corr_vals, rv_kms, rv_err_kms, obs_wave,
             fit_ok=False, popt=None, fwhm=np.nan,
             rv_min=-500.0, rv_max=500.0, save_dir="."):
    """
    Generate two CCF plots:
      Plot A : x-axis in km/s, limited to [rv_min, rv_max]
      Plot B : x-axis in Angstroms (Δλ = λ_ref * v/c)

    Parameters
    ----------
    lags_kms   : full lag array in km/s from template_correlate
    corr_vals  : CCF values
    rv_kms     : measured RV in km/s (Gaussian centre)
    rv_err_kms : 1-sigma RV uncertainty in km/s
    obs_wave   : observed wavelength array — used for λ_ref in Å plot
    fit_ok     : whether Gaussian fit succeeded
    popt       : Gaussian fit parameters [amp, mu, sigma, offset]
    fwhm       : FWHM of CCF peak in km/s
    rv_min/max : x-axis limits for km/s plot (default ±500)
    save_dir   : directory for output PNGs
    """

    def gaussian(x, amp, mu, sigma, offset):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

    # Crop to rv_min/rv_max window
    kms_mask  = (lags_kms >= rv_min) & (lags_kms <= rv_max)
    lags_crop = lags_kms[kms_mask]
    corr_crop = corr_vals[kms_mask]

    # Fine grid for Gaussian overlay
    rv_fine    = np.linspace(lags_crop[0], lags_crop[-1], 1000)
    gauss_fine = gaussian(rv_fine, *popt) if fit_ok and popt is not None else None

    # ── Plot A: km/s ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(lags_crop, corr_crop, color="steelblue", s=10,
               alpha=0.4, zorder=2, label="CCF data")

    if fit_ok and gauss_fine is not None:
        ax.plot(rv_fine, gauss_fine, color="red", lw=2.2, zorder=3,
                label=(f"Gaussian fit\n"
                       f"RV = {rv_kms:.2f} ± {rv_err_kms:.2f} km/s\n"
                       f"FWHM = {fwhm:.1f} km/s"))
        ax.fill_between(rv_fine, popt[3], gauss_fine,
                        color="red", alpha=0.10)
        ax.axvspan(rv_kms - rv_err_kms, rv_kms + rv_err_kms,
                   color="orange", alpha=0.20,
                   label=f"±1σ = ±{rv_err_kms:.2f} km/s")

    ax.axvline(rv_kms, color="red",  lw=1.5, ls="--", alpha=0.8)
    ax.axvline(0,      color="grey", lw=0.8, ls=":",  alpha=0.5,
               label="Zero velocity")

    ax.set_xlim(rv_min, rv_max)
    ax.set_xlabel("Radial Velocity (km/s)", fontsize=12)
    ax.set_ylabel("Correlation",            fontsize=12)
    ax.set_title(
        f"CCF — SVPer  |  RV = {rv_kms:.2f} ± {rv_err_kms:.2f} km/s",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path_kms = os.path.join(save_dir, "ccf_kms.png")
    plt.savefig(path_kms, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  CCF (km/s) saved → {path_kms}")

    # ── Plot B: Angstroms ─────────────────────────────────────────────────────
    # Convert velocity lags to wavelength shifts: Δλ = λ_ref * v/c
    c_kms   = c_light.to('km/s').value
    lam_ref = np.median(obs_wave)

    lags_aa   = lam_ref * lags_crop  / c_kms
    rv_aa     = lam_ref * rv_kms     / c_kms
    rv_err_aa = lam_ref * rv_err_kms / c_kms
    rv_min_aa = lam_ref * rv_min     / c_kms
    rv_max_aa = lam_ref * rv_max     / c_kms
    rv_fine_aa = lam_ref * rv_fine   / c_kms

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(lags_aa, corr_crop, color="steelblue", s=10,
               alpha=0.4, zorder=2, label="CCF data")

    if fit_ok and gauss_fine is not None:
        ax.plot(rv_fine_aa, gauss_fine, color="red", lw=2.2, zorder=3,
                label=(f"Gaussian fit\n"
                       f"Δλ = {rv_aa:.4f} ± {rv_err_aa:.4f} Å\n"
                       f"(= {rv_kms:.2f} ± {rv_err_kms:.2f} km/s)"))
        ax.fill_between(rv_fine_aa, popt[3], gauss_fine,
                        color="red", alpha=0.10)
        ax.axvspan(rv_aa - rv_err_aa, rv_aa + rv_err_aa,
                   color="orange", alpha=0.20,
                   label=f"±1σ = ±{rv_err_aa:.4f} Å")

    ax.axvline(rv_aa, color="red",  lw=1.5, ls="--", alpha=0.8)
    ax.axvline(0,     color="grey", lw=0.8, ls=":",  alpha=0.5,
               label="Zero shift")

    ax.set_xlim(rv_min_aa, rv_max_aa)
    ax.set_xlabel("Wavelength Shift Δλ (Å)", fontsize=12)
    ax.set_ylabel("Correlation",             fontsize=12)
    ax.set_title(
        f"CCF — SVPer  |  Δλ = {rv_aa:.4f} Å  "
        f"(λ_ref = {lam_ref:.2f} Å)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path_aa = os.path.join(save_dir, "ccf_angstrom.png")
    plt.savefig(path_aa, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  CCF (Angstrom) saved → {path_aa}")


# =============================================================================
# HELPER 4 — RV cross-correlation with uncertainty
# =============================================================================

def correct_rv_cross_correlation(obs_wave, obs_flux,
                                  synth_wave, synth_flux,
                                  noise_wave_lo  = None,
                                  noise_wave_hi  = None,
                                  fit_window_kms = 300.0,
                                  rv_min         = -500.0,
                                  rv_max         =  500.0,
                                  save_dir       = "."):
    """
    Measure RV via template cross-correlation with formal uncertainty.

    Noise is estimated directly from std(flux[line-free region]) —
    avoids the specutils WCS compatibility issue.

    RV uncertainty = fit covariance + noise-limited (Zucker 2003) in quadrature.

    Returns
    -------
    corrected_obs_wave : RV-corrected observed wavelengths (Å)
    rv_kms             : radial velocity (km/s)
    rv_err_kms         : 1-sigma uncertainty (km/s)
    """
    print("  Running template cross-correlation ...")

    # ── Noise estimation ──────────────────────────────────────────────────────
    if noise_wave_lo is None or noise_wave_hi is None:
        step    = np.median(np.diff(obs_wave))
        win_pix = max(int(5.0 / step), 10)
        best_std, best_lo, best_hi = np.inf, obs_wave[0], obs_wave[win_pix]
        for i in range(len(obs_flux) - win_pix):
            std = obs_flux[i : i + win_pix].std()
            if std < best_std:
                best_std = std
                best_lo  = obs_wave[i]
                best_hi  = obs_wave[i + win_pix]
        noise_wave_lo, noise_wave_hi = best_lo, best_hi
        print(f"  Auto noise region : {noise_wave_lo:.2f}–{noise_wave_hi:.2f} Å  "
              f"(std={best_std:.5f})")
    else:
        print(f"  Noise region : {noise_wave_lo:.2f}–{noise_wave_hi:.2f} Å")

    noise_mask  = (obs_wave >= noise_wave_lo) & (obs_wave <= noise_wave_hi)
    noise_value = obs_flux[noise_mask].std() if noise_mask.sum() >= 5 \
                  else obs_flux.std()
    snr         = 1.0 / noise_value if noise_value > 0 else np.nan
    print(f"  Per-pixel noise : {noise_value:.6f}   SNR ≈ {snr:.1f}")

    # ── Interpolate synthetic onto observed grid ───────────────────────────────
    synth_on_obs = np.interp(obs_wave, synth_wave, synth_flux,
                              left=1.0, right=1.0)
    obs_norm     = obs_flux    - np.median(obs_flux)
    synth_norm   = synth_on_obs - np.median(synth_on_obs)

    spec_axis = SpectralCoord(obs_wave * u.AA)

    obs_spec = Spectrum1D(
        spectral_axis = spec_axis,
        flux          = obs_norm * u.dimensionless_unscaled,
        uncertainty   = StdDevUncertainty(
            np.full_like(obs_norm, noise_value)
        )
    )
    synth_spec = Spectrum1D(
        spectral_axis = spec_axis,
        flux          = synth_norm * u.dimensionless_unscaled,
        uncertainty   = StdDevUncertainty(np.ones_like(synth_norm))
    )

    # ── Cross-correlate ────────────────────────────────────────────────────────
    correlation, lags = template_correlate(
        obs_spec, synth_spec,
        resample           = False,
        apodization_window = None
    )

    lags_kms  = lags.to(u.km / u.s).value
    corr_vals = correlation.value
    peak_idx  = np.argmax(corr_vals)
    rv_raw    = lags_kms[peak_idx]
    print(f"  Raw CCF peak : {rv_raw:.3f} km/s")

    # ── Gaussian fit ──────────────────────────────────────────────────────────
    win_mask = np.abs(lags_kms - rv_raw) <= fit_window_kms
    lags_win = lags_kms[win_mask]
    corr_win = corr_vals[win_mask]

    def gaussian(x, amp, mu, sigma, offset):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

    try:
        p0         = [corr_vals[peak_idx], rv_raw, 50.0, np.min(corr_win)]
        popt, pcov = curve_fit(gaussian, lags_win, corr_win,
                                p0=p0, maxfev=10000)
        rv_kms     = popt[1]
        fwhm       = 2.355 * abs(popt[2])
        fit_ok     = True

        rv_err_fit = np.sqrt(np.abs(pcov[1, 1]))
        n_pix      = win_mask.sum()
        rv_err_snr = fwhm / (snr * np.sqrt(n_pix))
        rv_err_kms = np.sqrt(rv_err_fit**2 + rv_err_snr**2)

        #print(f"  Gaussian fit RV  : {rv_kms:.3f} km/s")
        #print(f"  FWHM             : {fwhm:.2f} km/s")
        #print(f"  RV uncertainty   : ±{rv_err_kms:.3f} km/s  "
        #f"(fit=±{rv_err_fit:.3f}, SNR=±{rv_err_snr:.3f})")

    except Exception as e:
        print(f"  Gaussian fit failed ({e}) — using raw peak.")
        rv_kms     = rv_raw
        rv_err_kms = np.nan
        fwhm       = np.nan
        fit_ok     = False
        popt       = None

    if rv_kms > 0:
        print(f"\n  → SVPer receding    RV = +{rv_kms:.2f} ± {rv_err_kms:.2f} km/s")
    else:
        print(f"\n  → SVPer approaching RV =  {rv_kms:.2f} ± {rv_err_kms:.2f} km/s")

    # ── Generate both CCF plots ───────────────────────────────────────────────
    plot_ccf(
        lags_kms, corr_vals,
        rv_kms, rv_err_kms,
        obs_wave,
        fit_ok   = fit_ok,
        popt     = popt,
        fwhm     = fwhm,
        rv_min   = rv_min,
        rv_max   = rv_max,
        save_dir = save_dir
    )

    # ── Apply Doppler correction ───────────────────────────────────────────────
    doppler_factor     = 1.0 + rv_kms / c_light.to('km/s').value
    corrected_obs_wave = obs_wave / doppler_factor

    return corrected_obs_wave, rv_kms, rv_err_kms


# =============================================================================
# MAIN — spectral_synthesis
# =============================================================================

def spectral_synthesis(teff, logg, new_mh, v_turb, observed_data,
                       start_wav      = None,
                       end_wav        = None,
                       noise_wave_lo  = None,
                       noise_wave_hi  = None,
                       rv_min         = -500.0,
                       rv_max         =  500.0,
                       linelist_path  = "Downloads/ARES/moog_region1_cleaned.txt",
                       save_dir       = "."):
    """
    Full spectral synthesis pipeline:
      Step 1 — Continuum normalise observed spectrum
      Step 2 — Convolve synthetic to R=80,000
      Step 3 — Cross-correlate → Gaussian CCF → RV + uncertainty
               Produces ccf_kms.png and ccf_angstrom.png
      Step 4 — Plot observed vs synthetic + residuals

    Returns
    -------
    synth_data    : pymoog synth object
    r2            : R² of observed vs synthetic after RV correction
    obs_flux_norm : normalised observed flux array
    rv_kms        : radial velocity (km/s)
    rv_err_kms    : 1-sigma RV uncertainty (km/s)
    """
    os.makedirs(save_dir, exist_ok=True)

    if start_wav is None:
        start_wav = float(observed_data[:, 0][0])
    if end_wav is None:
        end_wav   = float(observed_data[:, 0][-1])

    print(f"\n{'='*56}")
    print(f"  Spectral Synthesis — SVPer")
    print(f"  Window : {start_wav:.1f} – {end_wav:.1f} Å")
    print(f"  Teff={teff:.0f}  logg={logg:.3f}  "
          f"[Fe/H]={new_mh:.4f}  vmicro={v_turb:.3f}")
    print(f"{'='*56}")

    # ── Generate synthetic spectrum ───────────────────────────────────────────
    synth_data = pymoog.synth.synth(
        teff, logg, 0,
        start_wav, end_wav, 80000,
        vmicro=v_turb, line_list='kurucz'
    )
    synth_data.prepare_file(
        model_format = 'kurucz',
        model_type   = 'kurucz',
        abun_change  = {26: new_mh},
        smooth_para  = ['m', 0, 0, 0, 20, 0]
    )
    synth_data.run_moog(output=False)
    synth_data.read_spectra()

    # ── Load and filter Fe linelist ───────────────────────────────────────────
    linelist = pd.read_csv(
        linelist_path, sep=r"\s+", header=None,
        names=["wavelength", "id", "EP", "loggf", "EW"]
    )
    linelist["id"] = linelist["id"].round(1)
    linelist["C6"] = 0.0
    linelist["D0"] = 0.0
    linelist = linelist[["wavelength", "id", "EP", "loggf", "C6", "D0", "EW"]]
    linelist = linelist.sort_values(by=["id", "wavelength"])

    plot_lines_df    = linelist[
        (linelist["wavelength"] >= start_wav) &
        (linelist["wavelength"] <= end_wav)
    ]
    element_map      = {26.0: "Fe I", 26.1: "Fe II"}
    labeled_lines_df = plot_lines_df[
        plot_lines_df["id"].isin(element_map.keys())
    ]

    fe1_n = (labeled_lines_df["id"] == 26.0).sum()
    fe2_n = (labeled_lines_df["id"] == 26.1).sum()
    print(f"  Fe I  lines in window : {fe1_n}")
    print(f"  Fe II lines in window : {fe2_n}")
    if fe2_n == 0:
        print("  ⚠ No Fe II lines — try a bluer window (4900–5500 Å)")

    # ── Mask observed data ────────────────────────────────────────────────────
    mask         = ((observed_data[:, 0] >= start_wav) &
                    (observed_data[:, 0] <= end_wav))
    raw_obs_wave = observed_data[:, 0][mask]
    raw_obs_flux = observed_data[:, 1][mask]

    if len(raw_obs_wave) < 10:
        raise ValueError(
            f"Only {len(raw_obs_wave)} pixels in [{start_wav}–{end_wav} Å]. "
            "Check start_wav/end_wav."
        )

    # ── Step 1: Continuum normalisation ───────────────────────────────────────
    print("\n[Step 1] Continuum normalisation ...")
    obs_flux_norm = local_normalise(raw_obs_wave, raw_obs_flux, n_knots=6)
    print(f"  Median normalised flux : {np.median(obs_flux_norm):.4f}  "
          "(ideal = 1.0)")

    # ── Step 2: Convolve synthetic ────────────────────────────────────────────
    print("\n[Step 2] Convolving synthetic spectrum ...")
    synth_flux_conv = convolve_to_resolution(
        synth_data.wav, synth_data.flux, resolution=80000
    )

    # ── Step 3: Cross-correlate → RV + CCF plots ──────────────────────────────
    print("\n[Step 3] RV measurement ...")
    rest_frame_obs_wave, rv_kms, rv_err_kms = correct_rv_cross_correlation(
        raw_obs_wave, obs_flux_norm,
        synth_data.wav, synth_flux_conv,
        noise_wave_lo  = noise_wave_lo,
        noise_wave_hi  = noise_wave_hi,
        fit_window_kms = 300.0,
        rv_min         = rv_min,
        rv_max         = rv_max,
        save_dir       = save_dir
    )

    # ── R² ────────────────────────────────────────────────────────────────────
    synth_on_obs = np.interp(
        rest_frame_obs_wave, synth_data.wav, synth_flux_conv,
        left=1.0, right=1.0
    )
    ss_res    = np.sum((obs_flux_norm - synth_on_obs) ** 2)
    ss_tot    = np.sum((obs_flux_norm - obs_flux_norm.mean()) ** 2)
    r2        = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    residuals = obs_flux_norm - synth_on_obs
    print(f"\n  R² = {r2:.4f}")

    # ── Step 4: Observed vs synthetic plot ────────────────────────────────────
    print("\n[Step 4] Plotting comparison ...")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(rest_frame_obs_wave, obs_flux_norm,
             color="black", lw=0.8, label="Observed (RV corrected)")
    ax1.plot(synth_data.wav, synth_flux_conv,
             color="red", lw=1.0, ls="--",
             label="Synthetic (convolved)")
    ax1.axhline(1.0, color="grey", lw=0.5, ls=":", alpha=0.5)
    ax1.set_ylabel("Normalized Flux", fontsize=11)
    ax1.set_ylim(0, 1.4)
    ax1.set_xlim(start_wav, end_wav)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(
        f"Observed vs Synthetic  |  "
        f"RV = {rv_kms:.2f} ± {rv_err_kms:.2f} km/s",
        fontsize=12
    )

    # Fe line markers
    for _, row in labeled_lines_df.iterrows():
        wav           = row["wavelength"]
        element_id    = row["id"]
        element_label = element_map[element_id]
        line_color    = "red" if element_id == 26.0 else "blue"
        ax1.axvline(wav, color=line_color, ls="--", alpha=0.4, lw=1.5)
        ax1.text(wav, 1.10, element_label, rotation=90,
                 color=line_color, fontsize=7,
                 ha="center", va="bottom")

    param_text = (
        f"Teff = {teff:.0f} K\n"
        f"log g = {logg:.2f}\n"
        f"[Fe/H] = {new_mh:.4f}\n"
        f"$v_t$ = {v_turb:.2f} km/s"
    )
    ax1.text(0.02, 0.02, param_text,
             transform=ax1.transAxes, fontsize=10,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="white", alpha=0.8))

    ax2.plot(rest_frame_obs_wave, residuals, color="black", lw=0.7)
    ax2.axhline(0,     color="grey",   lw=0.8, ls="--")
    ax2.axhline(+0.05, color="orange", lw=0.5, ls=":", alpha=0.7)
    ax2.axhline(-0.05, color="orange", lw=0.5, ls=":", alpha=0.7,
                label="±0.05")
    ax2.set_ylabel("Residuals",           fontsize=10)
    ax2.set_xlabel("Rest Wavelength (Å)", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.text(0.02, 0.90, f"$R^2$ = {r2:.4f}",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="white", alpha=0.8))

    plt.tight_layout()
    synth_path = os.path.join(save_dir, "spectral_synthesis_rv_corrected.png")
    plt.savefig(synth_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  Spectrum plot saved → {synth_path}")

    return synth_data, r2, obs_flux_norm, rv_kms, rv_err_kms


# =============================================================================
# EXAMPLE CALL
# =============================================================================

if __name__ == "__main__":

    observed_data = np.loadtxt(
        "/home/zaynarif/Downloads/HRS/svper_target_order.txt"
    )

    synth_data, r2, obs_norm, rv, rv_err = spectral_synthesis(
        teff          = 5055,
        logg          = 0.66,
        new_mh        = -0.107,
        v_turb        = 2.98,
        observed_data = observed_data,
        start_wav     = 6050,
        end_wav       = 6100,
        rv_min        = -500.0,
        rv_max        =  500.0,
        save_dir      = ".",
    )

    print(f"\nR²  = {r2:.4f}")
    print(f"RV  = {rv:.2f} ± {rv_err:.2f} km/s")


# In[ ]:


#RV correction code for extracted spectral order


# In[ ]:


import numpy as np
from astropy.constants import c as c_light

def apply_rv_correction(input_file, output_file, rv_kms):
    """
    Loads an observed spectrum, shifts the wavelengths to the rest frame
    using a known Radial Velocity, and saves the result to a new file.
    """
    print(f"Loading original spectrum: {input_file}")
    
    # 1. Load the observed spectrum (assuming col 0 is wave, col 1 is flux)
    data = np.loadtxt(input_file)
    obs_wave = data[:, 0]
    obs_flux = data[:, 1]
    
    # 2. Calculate the Doppler factor
    c_kms = c_light.to('km/s').value
    doppler_factor = 1.0 + (rv_kms / c_kms)
    
    # 3. Shift the wavelengths to the rest frame
    rest_frame_wave = obs_wave / doppler_factor
    
    # 4. Package the data back together side-by-side
    corrected_data = np.column_stack((rest_frame_wave, obs_flux))
    
    # 5. Save the new spectrum to a text file
    # Using a clean format: 4 decimal places for wavelength, 5 for flux
    np.savetxt(output_file, corrected_data, fmt="%10.4f  %10.5f", 
               header=f"RV Corrected by {rv_kms:.3f} km/s\nWavelength(Ang)  Flux", 
               comments="# ")
    
    print(f"Successfully shifted {len(rest_frame_wave)} wavelength points.")
    print(f"Corrected spectrum saved to: {output_file}\n")

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    # Define your paths and the exact RV you measured
    input_path  = "/home/zaynarif/Downloads/HRS/svper_target_order2.txt"
    output_path = "/home/zaynarif/Downloads/HRS/svper_target_order_rv_corr2_new.txt"
    
    measured_rv = 9.99  # Replace this with the exact RV output from your CCF code
    
    apply_rv_correction(input_path, output_path, measured_rv)


# In[ ]:


#Atmospheric parameter determination Code


# In[5]:


import os
import shutil
import numpy as np
import pandas as pd
import pymoog
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'qt')

# =============================================================================
# HELPER 1 — Run MOOG (With Automatic Folder Cleanup)
# =============================================================================
def run_moog(teff, logg, vmicro, linelist, i):
    rundir = f"./moog_run_{i}"
    os.makedirs(rundir, exist_ok=True)

    a = pymoog.abfind.abfind(
        teff=teff,
        logg=logg,
        m_h=0.0,
        line_list=linelist,
        vmicro=vmicro
    )

    a.prepare_file(model_type='kurucz', rundir_path=rundir + "/")
    a.run_moog()
    a.read_output()

    # --- CLEANUP: Delete the temporary folder now that we have the data ---
    if os.path.exists(rundir):
        shutil.rmtree(rundir)

    return a

# =============================================================================
# HELPER 2 — Compute Slopes
# =============================================================================
def compute_slopes(a):
    fe1 = a.abfind_res[26.0]
    
    # Safety check in case a tight cut removes all Fe II lines
    if 26.1 in a.abfind_res:
        fe2 = a.abfind_res[26.1]
        fe2_mean = fe2["abund"].mean()
    else:
        fe2_mean = np.nan

    # Slopes
    slope_EP = np.polyfit(fe1["EP"], fe1["abund"], 1)[0]
    slope_RW = np.polyfit(fe1["logRWin"], fe1["abund"], 1)[0]

    # Ionization balance
    fe1_mean = fe1["abund"].mean()
    
    fe_diff = fe1_mean - fe2_mean if not np.isnan(fe2_mean) else 0.0

    print("fe I mean:", fe1_mean)
    print("fe II mean:", fe2_mean)

    return slope_EP, slope_RW, fe_diff, fe1_mean, fe2_mean

# =============================================================================
# HELPER 3 — Generate 3-Panel Diagnostics Plot
# =============================================================================
def plot_final_diagnostics(a_final, condition_name=""):
    fe1 = a_final.abfind_res.get(26.0, pd.DataFrame())
    fe2 = a_final.abfind_res.get(26.1, pd.DataFrame())
    
    n_fe1 = len(fe1)
    n_fe2 = len(fe2)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -------- Panel 1: Abundance vs EP --------
    if n_fe1 > 0:
        x_ep = fe1["EP"]
        y_ep = fe1["abund"]
        slope_ep, int_ep = np.polyfit(x_ep, y_ep, 1)
        
        axes[0].scatter(x_ep, y_ep, color='tab:blue')
        axes[0].set_title(f"Abundance vs EP ({n_fe1} Fe I lines)\nSlope={slope_ep:.5f}")
    
    axes[0].set_xlabel("EP (eV)")
    axes[0].set_ylabel("Abundance")
    axes[0].grid(True)

    # -------- Panel 2: Abundance vs RW --------
    if n_fe1 > 0:
        x_rw = fe1["logRWin"]
        y_rw = fe1["abund"]
        slope_rw, int_rw = np.polyfit(x_rw, y_rw, 1)
        
        axes[1].scatter(x_rw, y_rw, color='tab:blue')
        axes[1].set_title(f"Abundance vs RW ({n_fe1} Fe I lines)\nSlope={slope_rw:.5f}")

    axes[1].set_xlabel("log(RW)")
    axes[1].set_ylabel("Abundance")
    axes[1].grid(True)

    # -------- Panel 3: Fe I vs Fe II & Sigmas --------
    if n_fe1 > 0:
        axes[2].scatter(fe1["EP"], fe1["abund"], label=f"Fe I (N={n_fe1})", color='tab:blue')
        mean_fe1 = fe1["abund"].mean()
        std_fe1 = fe1["abund"].std()
        
        axes[2].axhline(mean_fe1, color='tab:blue', linestyle='--', label='Mean')
        axes[2].axhline(mean_fe1 + 2*std_fe1, color='tab:blue', linestyle=':', label='±2σ')
        axes[2].axhline(mean_fe1 - 2*std_fe1, color='tab:blue', linestyle=':')
        axes[2].axhline(mean_fe1 + 3*std_fe1, color='tab:blue', linestyle='-.', label='±3σ')
        axes[2].axhline(mean_fe1 - 3*std_fe1, color='tab:blue', linestyle='-.')
        
    if n_fe2 > 0:
        axes[2].scatter(fe2["EP"], fe2["abund"], label=f"Fe II (N={n_fe2})", color='tab:orange')

    axes[2].set_xlabel("EP (eV)")
    axes[2].set_ylabel("Abundance")
    axes[2].set_title(f"Fe I vs Fe II\n{condition_name}")
    axes[2].legend()
    axes[2].grid(True)

    #for ax in axes:
        #ax.set_ylim(2, 9)

    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN OPTIMIZER
# =============================================================================
def optimize_params(teff, logg, vmicro, linelist, max_iter=150):
    ep_list, rw_list, fe_list = [], [], []
    teff_list, logg_list, vmicro_list = [], [], []
    stable_count = 0  

    for i in range(max_iter):
        a = run_moog(teff, logg, vmicro, linelist, i)
        slope_EP, slope_RW, fe_diff, fe1_mean, fe2_mean = compute_slopes(a)

        print(f"\nIteration {i+1}")
        print(f"Teff={teff:.1f}, logg={logg:.2f}, vmicro={vmicro:.2f}")
        print(f"EP slope={slope_EP:.5f}")
        print(f"RW slope={slope_RW:.5f}")
        print(f"Fe diff={fe_diff:.5f}")

        ep_list.append(slope_EP)
        rw_list.append(slope_RW)
        fe_list.append(fe_diff)
        teff_list.append(teff)
        logg_list.append(logg)
        vmicro_list.append(vmicro)

        if abs(slope_EP) <= 0.01 and abs(slope_RW) <= 0.01 and abs(fe_diff) <= 0.01:
            stable_count += 1
            print(f"Parameters stable. Consecutive stable iterations: {stable_count}/5")
            if stable_count >= 5:
                print(f"*** Convergence reached after {i+1} iterations! ***")
                break
        else:
            stable_count = 0

        # Update
        if abs(slope_EP) > 0.01:
            teff += slope_EP * 50
        if abs(slope_RW) > 0.01:
            vmicro += slope_RW * 0.2
        if abs(fe_diff) > 0.01:
            logg += fe_diff * 0.1

        # Clamp
        teff = np.clip(teff, 4800, 6000)
        logg = np.clip(logg, 0.5, 2.0)
        vmicro = np.clip(vmicro, 2.0, 4.0)

    return teff, logg, vmicro, ep_list, rw_list, fe_list, teff_list, logg_list, vmicro_list

# =============================================================================
# SMART LINELIST LOADER
# =============================================================================
def load_and_verify_linelist(filepath):
    print(f"\nLoading linelist from: {filepath}")
    
    # Let pandas read the file and count the columns automatically
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    
    if len(df.columns) == 5:
        print("Detected 5-column format. Adding C6 and D0...")
        df.columns = ["wavelength", "id", "EP", "loggf", "EW"]
        df["C6"] = 0.0
        df["D0"] = 0.0
    elif len(df.columns) == 7:
        print("Detected 7-column format.")
        df.columns = ["wavelength", "id", "EP", "loggf", "C6", "D0", "EW"]
    else:
        raise ValueError(f"File has {len(df.columns)} columns. Expected 5 or 7.")
        
    # Reorder to strict MOOG standard
    df = df[["wavelength", "id", "EP", "loggf", "C6", "D0", "EW"]]
    
    # Force everything to be numeric (this fixes any hidden NaN or string issues)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    print(f"Successfully loaded {len(df)} lines.")
    print("-" * 50)
    print(df.head(3)) # Show a quick preview to ensure columns align!
    print("-" * 50)
    
    return df.sort_values(by=["id", "wavelength"])

# =============================================================================
# EXECUTION SCRIPT
# =============================================================================
if __name__ == "__main__":
    
    # 1. Use the smart loader on your precleaned file
    PRECLEANED_FILE = "/home/zaynarif/Downloads/ARES/moog_region1_new.txt"
    master_linelist = load_and_verify_linelist(PRECLEANED_FILE)

    # 2. Define conditions
    test_conditions = [
         #{"max_ew": 200, "min_ep": 2.0, "max_ep": 6.0},
          {"max_ew": 200, "min_ep": 2.5, "max_ep": 6.0},
         #{"max_ew": 150, "min_ep": 2.0, "max_ep": 6.0},
         #{"max_ew": 150, "min_ep": 2.5, "max_ep": 6.0}
    ]

    print("\nStarting Robustness Tests for SV Persei...\n")
    print("=" * 60)

    summary_results = []

    # 3. Loop through conditions
    for cond in test_conditions:
        # Apply cuts
        filtered_linelist = master_linelist[
            (master_linelist["EW"] < cond["max_ew"]) &
            (master_linelist["EP"] > cond["min_ep"]) &
            (master_linelist["EP"] < cond["max_ep"])
        ].copy()
        
        condition_str = f"EW < {cond['max_ew']} mA | EP > {cond['min_ep']} eV | EP < {cond['max_ep']} eV"
        print(f"\nTesting Condition: {condition_str}")
        print(f"Lines remaining after cut: {len(filtered_linelist)}")
        
        # Failsafe: Don't run optimizer if lines were wiped out
        if len(filtered_linelist) == 0:
            print("ERROR: Cut removed all lines. Skipping this condition.")
            print("-" * 60)
            continue
            
        # Run optimizer
        teff_final, logg_final, vmicro_final, *_ = optimize_params(
            5000, 0.4, 2.8, filtered_linelist
        )
        
        print(f"\nGenerating plots for {condition_str}...")
        a_final = run_moog(teff_final, logg_final, vmicro_final, filtered_linelist, "plot_run")
        
        # ---> GRAB THE FINAL LINE COUNTS HERE <---
        n_fe1 = len(a_final.abfind_res.get(26.0, []))
        n_fe2 = len(a_final.abfind_res.get(26.1, []))
        
        summary_results.append({
            "EW_cut": cond["max_ew"],
            "EP_cut": cond["min_ep"],
            "EPm_cut": cond["max_ep"],
            "Teff": teff_final,
            "logg": logg_final,
            "vt": vmicro_final,
            "n_fe1": n_fe1,
            "n_fe2": n_fe2
        })
        
        plot_final_diagnostics(a_final, condition_name=condition_str)
        
        print("-" * 60)

    # Print Final Summary Table
    if summary_results:
        print("\nFINAL ROBUSTNESS SUMMARY:")
        # Expanded table headers to include line counts
        print(f"{'Condition':<25} | {'Teff (K)':<8} | {'log g':<6} | {'v_micro':<7} | {'Fe I':<5} | {'Fe II':<5}")
        print("-" * 73)
        for res in summary_results:
            cond_str = f"EW<{res['EW_cut']}, EP>{res['EP_cut']}, EP<{res['EPm_cut']}"
            print(f"{cond_str:<25} | {res['Teff']:<8.1f} | {res['logg']:<6.2f} | {res['vt']:<7.2f} | {res['n_fe1']:<5} | {res['n_fe2']:<5}")


# In[ ]:


#Synthetic spectra code


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymoog
get_ipython().run_line_magic('matplotlib', 'qt')

observed_data = np.loadtxt("/home/zaynarif/Downloads/HRS/svper_target_order_rv_corr.txt")

# Add this right after you load the new file
print(f"Order 1 ranges from {observed_data[0, 0]:.1f} Å to {observed_data[-1, 0]:.1f} Å")

def spectral_synthesis(teff, logg, new_mh, v_turb, observed_data):

    # Define synthesis region
    start_wav = 6001.5
    end_wav = 6222.7

    # Generate synthetic spectrum
    synth_data = pymoog.synth.synth(
        teff,
        logg,
        0,
        start_wav,
        end_wav,
        80000,
        vmicro=v_turb,
        line_list='kurucz'
    )

    synth_data.prepare_file(
        model_format='kurucz',
        model_type='kurucz',
        abun_change={26: new_mh},
        smooth_para=['m', 0, 0, 0, 20, 0]
    )

    synth_data.run_moog(output=True)
    synth_data.read_spectra()

    # Load Fe linelist
    linelist = pd.read_csv(
        "Downloads/ARES/moog_region1_cleaned.txt",
        sep=r"\s+",
        header=None,
        names=["wavelength","id","EP","loggf","EW"]
    )
    linelist["id"] = linelist["id"].round(1)
    linelist["C6"] = 0.0
    linelist["D0"] = 0.0
    linelist = linelist[["wavelength","id","EP","loggf","C6","D0","EW"]]
    linelist = linelist.sort_values(by=["id","wavelength"])

    print(linelist.head())

    # Select lines in synthesis region
    plot_lines_df = linelist[
        (linelist["wavelength"] >= start_wav) &
        (linelist["wavelength"] <= end_wav)
    ]

    # Map element IDs
    element_map = {
        26.0: "Fe I",
        26.1: "Fe II"
    }

    labeled_lines_df = plot_lines_df[
        plot_lines_df["id"].isin(element_map.keys())
    ]

# Mask observed spectrum to match region
    mask = (observed_data[:,0] >= start_wav) & (observed_data[:,0] <= end_wav)

    # --- NEW: Local Continuum Correction ---
    # Find the 95th percentile of the flux in this specific window to represent the continuum
    local_continuum = np.percentile(observed_data[:,1][mask], 95)
    
    # Scale the observed flux up so the continuum sits at 1.0
    corrected_obs_flux = observed_data[:,1][mask] / local_continuum
    # ---------------------------------------

    # ----------- PLOT -----------

    fig, ax1 = plt.subplots(figsize=(10,6))

    # Observed spectrum
    ax1.plot(
        observed_data[:,0][mask],
        corrected_obs_flux,  # <-- Use the scaled flux here
        color="black",
        label="Observed spectrum"
    )

    # Synthetic spectrum
    ax1.plot(
        synth_data.wav,
        synth_data.flux,
        "--",
        color="red",
        label="Synthetic spectrum"
    )

    ax1.set_ylabel("Normalized Flux")
    ax1.set_xlabel("Wavelength (Å)")
    ax1.set_ylim(0, 1.4)
    ax1.set_xlim(start_wav, end_wav)

    ax1.legend(loc="upper right")

    # Plot Fe line markers
# Plot Fe line markers
    for _, row in labeled_lines_df.iterrows():
        wav = row["wavelength"]
        element_id = row["id"]
        element_label = element_map[element_id]

        # Assign colors: Red for neutral (Fe I), Blue for ionized (Fe II)
        line_color = "red" if element_id == 26.0 else "blue"

        # Vertical line
        ax1.axvline(
            wav,
            color=line_color,
            linestyle="--", # Changed to dashed for better visibility
            alpha=0.5,
            lw=2.0
        )

        # Text label
        ax1.text(
            wav,
            1.08, # Moved slightly higher to avoid clipping
            element_label,
            rotation=90,
            color=line_color,
            fontsize=8,
            ha="center",
            va="bottom"
        )

    # Stellar parameter text
    param_text = (
        f"Teff = {teff:.0f} K\n"
        f"log g = {logg:.2f}\n"
        f"[Fe/H] = {new_mh:.2f}\n"
        f"$v_t$ = {v_turb:.2f} km/s"
    )

    ax1.text(
        0.02,
        0.02,
        param_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    plt.title("Observed vs Synthetic Spectrum")
    plt.tight_layout()
    plt.show()


# In[6]:


spectral_synthesis(5055, 0.66, -0.10, 2.89, observed_data)


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymoog
get_ipython().run_line_magic('matplotlib', 'qt')

observed_data = np.loadtxt("/home/zaynarif/Downloads/HRS/svper_target_order_rv_corr.txt")

# Add this right after you load the new file
print(f"Order 1 ranges from {observed_data[0, 0]:.1f} Å to {observed_data[-1, 0]:.1f} Å")

def spectral_synthesis(teff, logg, new_mh, v_turb, observed_data):

    # Define synthesis region
    start_wav = 6001.5
    end_wav = 6222.7

    # Generate synthetic spectrum
    synth_data = pymoog.synth.synth(
        teff,
        logg,
        0,
        start_wav,
        end_wav,
        80000,
        vmicro=v_turb,
        line_list='kurucz'
    )

    synth_data.prepare_file(
        model_format='kurucz',
        model_type='kurucz',
        abun_change={26: new_mh},
        smooth_para=['m', 0, 0, 0, 20, 0]
    )

    synth_data.run_moog(output=True)
    synth_data.read_spectra()

    # Load Fe linelist
    linelist = pd.read_csv(
        "Downloads/ARES/moog_region1_cleaned.txt",
        sep=r"\s+",
        header=None,
        names=["wavelength","id","EP","loggf","EW"]
    )
    linelist["id"] = linelist["id"].round(1)
    linelist["C6"] = 0.0
    linelist["D0"] = 0.0
    linelist = linelist[["wavelength","id","EP","loggf","C6","D0","EW"]]
    linelist = linelist.sort_values(by=["id","wavelength"])

    print(linelist.head())

    # Select lines in synthesis region
    plot_lines_df = linelist[
        (linelist["wavelength"] >= start_wav) &
        (linelist["wavelength"] <= end_wav)
    ]

    # Map element IDs
    element_map = {
        26.0: "Fe I",
        26.1: "Fe II"
    }

    labeled_lines_df = plot_lines_df[
        plot_lines_df["id"].isin(element_map.keys())
    ]

    # Mask observed spectrum to match region
    mask = (observed_data[:,0] >= start_wav) & (observed_data[:,0] <= end_wav)

    # --- NEW: Local Continuum Correction ---
    # Find the 95th percentile of the flux in this specific window to represent the continuum
    local_continuum = np.percentile(observed_data[:,1][mask], 95)
    
    # Scale the observed flux up so the continuum sits at 1.0
    corrected_obs_flux = observed_data[:,1][mask] / local_continuum
    # ---------------------------------------

    # ----------- PLOT -----------

    fig, ax1 = plt.subplots(figsize=(10,6))

    # Observed spectrum
    ax1.plot(
        observed_data[:,0][mask],
        corrected_obs_flux,  # <-- Use the scaled flux here
        color="black",
        label="Observed spectrum"
    )

    # Synthetic spectrum
    ax1.plot(
        synth_data.wav,
        synth_data.flux,
        "--",
        color="red",
        label="Synthetic spectrum"
    )

    ax1.set_ylabel("Normalized Flux")
    ax1.set_xlabel("Wavelength (Å)")
    ax1.set_ylim(0, 1.4)
    ax1.set_xlim(start_wav, end_wav)

    ax1.legend(loc="upper right")

    # Plot Fe line markers
    for _, row in labeled_lines_df.iterrows():
        wav = row["wavelength"]
        element_id = row["id"]
        element_label = element_map[element_id]

        # Assign colors: Red for neutral (Fe I), Blue for ionized (Fe II)
        line_color = "red" if element_id == 26.0 else "blue"

        # Vertical line
        ax1.axvline(
            wav,
            color=line_color,
            linestyle="--", 
            alpha=0.5,
            lw=2.0
        )

        # Text label
        ax1.text(
            wav,
            1.08, 
            element_label,
            rotation=90,
            color=line_color,
            fontsize=8,
            ha="center",
            va="bottom"
        )

    # Stellar parameter text
    param_text = (
        f"Teff = {teff:.0f} K\n"
        f"log g = {logg:.2f}\n"
        f"[Fe/H] = {new_mh:.4f}\n"  # <--- Changed .2f to .4f to preserve the precision
        f"$v_t$ = {v_turb:.2f} km/s"
    )

    ax1.text(
        0.02,
        0.02,
        param_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    plt.title("Observed vs Synthetic Spectrum")
    plt.tight_layout()
    plt.show()

# Call the function passing -0.1070 for the metallicity parameter
spectral_synthesis(5055, 0.66, -0.1070, 2.89, observed_data)


# In[ ]:




