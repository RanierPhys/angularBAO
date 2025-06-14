# Cosmological Analysis from Angular BAO Data

Likelihood implementation for cosmological parameter estimation using angular Baryon Acoustic Oscillation (BAO) data [_Ranier Menote , Valerio Marra_](https://academic.oup.com/mnras/article/513/2/1600/6556012), with optional combination with Supernova (SNe) data.



---

## Overview

The Python scripts (e.g., `Angular_BAO.py`, `BAO_Sn_flat_prior.py`) perform Markov Chain Monte Carlo (MCMC) sampling assuming a flat ΛCDM cosmological model. The parameter space explored includes:

- `(r_d * h, Ω_m)` — when using angular BAO data only.
- `(r_d * h, Ω_m, M)` — when combining angular BAO with Supernova data.

---

## Data Files

### `Final_Table.txt`

This file contains the angular BAO measurements:

| Column | Description |
|--------|-------------|
| 1st    | Redshift (z) |
| 2nd    | Angular BAO scale (biased) |
| 3rd    | Angular BAO scale (unbiased) |
| 4th    | Statistical error |
| 5th    | Systematic error |
| 6th    | Total error: sqrt(stat² + sys²) |
| 7th    | Percent error relative to the unbiased measurement |

### `Cov_BAO.txt`

Covariance matrix for the angular BAO measurements (same ordering as above).

### `Snv.txt`

Pantheon Supernova dataset. Used to combine SNe Ia likelihood with angular BAO for tighter parameter constraints.

### `Snvsys.txt`

Systematic error covariance matrix for the Supernova measurements.

---

## Script: `Angular_BAO.py`

This script uses **only angular BAO data** to infer cosmological parameters. The likelihood is built using the following chi-squared function:


where:

- `θ_BAO(z) = r_d / [(1 + z) * d_A(z)]`
- `d_A(z)` is the angular diameter distance

Because the likelihood depends on the ratio `r_d / d_A(z)`, there is a degeneracy between `r_d` and `H0`. Thus, the parameter space becomes `(r_d * h, Ω_m)`.

Flat priors are assumed on the parameters unless specified otherwise.

---

## Script: `BAO_Sn_flat_prior.py`

This script **combines angular BAO and Supernova data**. The chi-squared function for Supernovae is:


where:

- `M` is the absolute magnitude of supernovae
- `m(z) = 5 * log10(D_L(z) / 10 pc) + M`
- `D_L(z)` is the luminosity distance

The total likelihood is:


With flat priors, the degeneracy between `r_d` and `H0` remains. Thus, the parameter space is `(r_d * h, Ω_m, M)`. This degeneracy can be broken by using an informative prior on `M` (e.g., from Cepheid-calibrated SNe).

---

## Requirements

- Python 3.x
- `numpy`, `scipy`, `matplotlib`
- Optional: `emcee` for MCMC sampling

To install dependencies:

```bash
pip install numpy scipy matplotlib emcee


