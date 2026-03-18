# Supplementary Code: "Measuring Primitive Accumulation: An Information-Theoretic Approach to Capitalist Enclosure in PIK2, Indonesia"

[![arXiv](https://img.shields.io/badge/arXiv-2603.13715-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.13715)
[![DOI](https://zenodo.org/badge/1176348788.svg)](https://doi.org/10.5281/zenodo.18947822)
![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![netCDF4](https://img.shields.io/badge/netCDF4-005571?style=for-the-badge)
![Rasterio](https://img.shields.io/badge/Rasterio-4B8BBE?style=for-the-badge)
![PyProj](https://img.shields.io/badge/PyProj-5C5C5C?style=for-the-badge)
![PyGMT](https://img.shields.io/badge/PyGMT-2980B9?style=for-the-badge)
[![License: MIT](https://img.shields.io/badge/License-MIT-F1C40F?style=for-the-badge)](LICENSE)
[![OSF DOI](https://img.shields.io/badge/OSF_DOI-10.17605%2FOSF.IO%2FZTWVU-337AB7?style=for-the-badge)](https://doi.org/10.17605/OSF.IO/ZTWVU)

## Overview

This repository contains the data processing pipelines and analytical scripts required to reproduce the quantitative modeling and figures presented in the manuscript *"Measuring Primitive Accumulation: An Information-Theoretic
Approach to Capitalist Enclosure in PIK2, Indonesia"*.

The codebase operationalises Marxian political economy concepts (primitive accumulation, subsumption, and enclosure) through the application of information geometry, absorbing Markov chains, and percolation theory applied to high-resolution Sentinel-2 Land Use/Land Cover (LULC) data (2017–2024).

## Authors

* **Sandy H. S. Herho** 
* **Alfita P. Handayani** (Corresponding author: alfita@itb.ac.id)
* **Karina A. Sujatmiko**
* **Faruq Khadami**
* **Iwan P. Anwar**


## Data Availability

The raw input GeoTIFFs (Esri Sentinel-2 10-m LULC) and the processed outputs (NetCDF4 grids, statistical reports, and figures) are too large for GitHub and are hosted on the Open Science Framework (OSF).

**OSF Repository:** [https://doi.org/10.17605/OSF.IO/ZTWVU](https://doi.org/10.17605/OSF.IO/ZTWVU)

All data hosted on OSF is released under the MIT License. Please download the dataset and place the contents in the appropriate `geotiff/` and `netcdf/` directories before running the pipeline.

## Mathematical Framework

The analytical scripts contained in this repository construct several rigorous mathematical spaces to quantify land enclosure. Key formalisms include:

### 1. Information Geometry on the Probability Simplex

To measure the "velocity" of landscape transformation, we calculate the Fisher-Rao geodesic distance on the probability simplex. For two class probability distributions $p$ and $q$, the distance represents the minimal path length under the Fisher information metric:

$$d_{\mathrm{FR}}(p,q) = 2 \arccos\left(\sum_{i=1}^{N} \sqrt{p_i q_i}\right)$$

Kullback-Leibler (KL) divergence is also evaluated to measure the relative entropy and directionality of transitions between consecutive states:

$$D_{\mathrm{KL}}(p \parallel q) = \sum_{i=1}^{N} p_i \ln\left(\frac{p_i}{q_i}\right)$$

We complement this with Shannon entropy $H(p)$ to track the pre-enclosure diversification phase, and the Rényi entropy spectrum $H_\alpha(p)$ to evaluate multi-scale spatial organisation and dominance:

$$H(p) = -\sum_{i=1}^{N} p_i \ln(p_i)$$

$$H_\alpha(p) = \frac{1}{1-\alpha} \ln\left(\sum_{i=1}^{N} p_i^\alpha\right)$$

To assess the monotonic trend of the entropy trajectory over time, a non-parametric Mann-Kendall test is applied.

### 2. Markovian Dynamics and Absorbing Chains

The landscape transitions are modeled as a discrete-time Markov chain. The empirical transition probabilities and their asymptotic standard errors are given by:

$$p_{ij} = \frac{n_{ij}}{\sum_k n_{ik}}, \quad SE(p_{ij}) = \sqrt{\frac{p_{ij}(1-p_{ij})}{n_i}}$$

Capitalist spatial accumulation is modeled as an absorbing process, where *Built Area* represents the forced absorbing state ($P_{BB} = 1$). The fundamental matrix $\mathbf{N}$ is derived from the transient sub-matrix $\mathbf{Q}$:

$$\mathbf{N} = (\mathbf{I} - \mathbf{Q})^{-1}$$

The expected time to absorption $\mathbb{E}[\mathbf{T}]$ (representing the "speed of enclosure") from any transient state, and its intrinsic variance $\mathrm{Var}[\mathbf{T}]$ (representing the spread of the first-passage-time distribution), are formulated as:

$$\mathbb{E}[\mathbf{T}] = \mathbf{N} \mathbf{1}$$

$$\mathrm{Var}[\mathbf{T}] = (2\mathbf{N} - \mathbf{I})\mathbb{E}[\mathbf{T}] - (\mathbb{E}[\mathbf{T}])^2$$

The non-stationarity of the transition matrices over time is formally evaluated using a $G$-test (log-likelihood ratio) and Frobenius norms.

### 3. Percolation Theory and Fractal Dimension

To mathematically determine whether spatial expansion represents a planned supercritical phenomenon or random colonisation, we apply percolation theory. The order parameter is defined as the fraction of built pixels contained within the largest connected cluster ($S_{\max}$):

$$\text{Order Parameter} = \frac{S_{\max}}{S_{\mathrm{built}}}$$

The morphological irregularity of the expanding capitalist frontier is quantified using the box-counting fractal dimension $d_f$. It is derived via ordinary least squares regression on the cluster boundary:

$$\ln N(\epsilon) \sim -d_f \ln \left(\frac{1}{\epsilon}\right)$$

where $N(\epsilon)$ is the number of boxes of size $\epsilon$ required to cover the boundary of the largest connected component.

### 4. Marxian Ternary Simplex Aggregation

For the macroeconomic analysis of structural subsumption, detailed LULC classes are aggregated into a 3-simplex representing the Marxian triad (excluding the ocean):

$$p_{\text{commons}} + p_{\text{agrarian}} + p_{\text{capital}} = 1$$

Fisher-Rao distances, accelerations, and trajectory sinuosity (arc length divided by direct displacement) are re-computed on this sub-manifold to track the secular progression of enclosure.

## Repository Structure

```text
.
├── geotiff/                  # Raw input Esri Sentinel-2 10-m LULC GeoTIFF files (Download from OSF)
├── netcdf/                   # Processed NetCDF4 output directory
├── scripts/                  # Analytical Python scripts
│   ├── extractGeotiff.py
│   ├── informationEntropy.py
│   ├── map.py
│   ├── markovTransition.py
│   ├── percolation.py
│   ├── plotLULC.py
│   └── simplex.py
├── figs/                     # Output directory for generated figures (.pdf, .png)
├── reports/                  # Output directory for statistical reports (.txt)
├── .gitignore
└── LICENSE                   # MIT License

```

## Execution Pipeline

The scripts are designed to be run sequentially from within the `scripts/` directory.

### 1. Data Preparation

* `extractGeotiff.py`: Parses raw Sentinel-2 GeoTIFFs, projects them to a uniform WGS-84 grid, filters classes, and compiles an annual time-series into a compressed `pik2LULC.nc` NetCDF4 file.

### 2. Spatial Context and Visualisation

* `map.py`: Generates the primary geographic study area map utilizing PyGMT.
* `plotLULC.py`: Renders the 2×4 panel publication map of annual LULC classifications.

### 3. Mathematical Analyses

* `informationEntropy.py`: Executes the information-theoretic framework (Fisher-Rao, Kullback-Leibler, Rényi spectra).
* `markovTransition.py`: Constructs the empirical transition matrices and formulates the absorbing Markov chain.
* `percolation.py`: Applies percolation theory to evaluate spatial continuity and fractal dimension.
* `simplex.py`: Aggregates detailed LULC classes into a macro-sociological Ternary Simplex (Commons, Agrarian, Capital).

## Copyright & License

Copyright © 2026 Center for Agrarian Studies, Bandung Institute of Technology (ITB).

This project, including the code in this repository and the associated data hosted on OSF, is licensed under the [MIT License](https://github.com/sandyherho/supplPIK2LULC/blob/main/LICENSE).
