# Cosmological Analysis from Angular BAO Data

Likelihood implementation for cosmological parameter estimation using angular Baryon Acoustic Oscillation (BAO) data, with optional combination with Supernova (SNe) data.

## Overview

The Python scripts (e.g., `Angular_BAO.py`, `BAO_Sn_flat_prior.py`) perform Markov Chain Monte Carlo (MCMC) sampling assuming a flat $\Lambda$CDM cosmological model. The parameter space explored includes either:

- $(r_d h, \Omega_m)$ — when angular BAO data is used alone.
- $(r_d h, \Omega_m, M)$ — when angular BAO is combined with Supernova data.

---

## Data Files

### `Final_Table.txt`

This file contains the angular BAO measurements as described in the paper *[insert reference or link here]*:

| Column | Description |
|--------|-------------|
| 1st    | Redshift ($z$) |
| 2nd    | Angular BAO scale (biased measurement) |
| 3rd    | Angular BAO scale (unbiased measurement) |
| 4th    | Statistical error |
| 5th    | Systematic error |
| 6th    | Total error = $\sqrt{\text{stat}^2 + \text{sys}^2}$ |
| 7th    | Percentage of total error relative to the unbiased measurement |

### `Cov_BAO.txt`

Covariance matrix for the angular BAO measurements (same ordering as `Final_Table.txt`).

### `Snv.txt`

Pantheon Supernova data used to include SNe Ia likelihood in the analysis to improve constraints on cosmological parameters.

### `Snvsys.txt`

Covariance matrix for the systematic errors in the Supernova measurements.

---

## Script: `Angular_BAO.py`

This script performs parameter inference using only angular BAO data. As described in *[insert reference here]*, the chi-squared function used is:

$$
\chi^2_{\text{BAO}}(H_0, \Omega_m, r_d) = \sum_{ij} [\theta_{BAO}^{\text{th}}(z_i) - \theta_{BAO,i}^{\text{obs}}] \, \Sigma^{-1}_{ij} \, [\theta_{BAO}^{\text{th}}(z_j) - \theta_{BAO,j}^{\text{obs}}]
$$

where:

- $\theta_{BAO}(z) = \dfrac{r_d}{(1+z) \, d_A(z)}$
- $d_A(z)$ is the angular diameter distance.

Due to the degeneracy in the ratio $\dfrac{r_d}{d_A}$, the analysis focuses on the combined parameter $r_d h$ rather than $H_0$ and $r_d$ separately. The sampling space is $(r_d h, \Omega_m)$.

Flat priors are assumed on all parameters unless otherwise specified.

---

## Script: `BAO_Sn_flat_prior.py`

This script combines angular BAO and Supernova data to perform joint cosmological parameter estimation.

The chi-squared for Supernova data is defined as:

$$
\chi^2_{\text{SNe}}(H_0, \Omega_m, M) = \sum_{ij} [m^{\text{th}}(z_i) - m_i^{\text{obs}}] \, \Sigma^{-1}_{ij} \, [m^{\text{th}}(z_j) - m_j^{\text{obs}}]
$$

where:

- $M$ is the absolute magnitude of the Supernovae.
- $m(z) = 5 \log_{10} \left( \dfrac{D_L(z)}{10\,\text{pc}} \right) + M$ is the apparent magnitude.
- $D_L(z)$ is the luminosity distance.

The total likelihood is:

$$
\chi^2 = \chi^2_{\text{BAO}} + \chi^2_{\text{SNe}}
$$

With only flat priors, the degeneracy between $H_0$ and $r_d$ persists. Therefore, the parameter space becomes $(r_d h, \Omega_m, M)$. This degeneracy can be broken by applying an informative prior on $M$ from external calibrations (e.g., Cepheid-calibrated SNe).

---

## Requirements

- Python 3.x
- `numpy`, `scipy`, `matplotlib`
- Optional: `emcee` for MCMC sampling

Install via pip:

```bash
pip install numpy scipy matplotlib emcee
