# Supplementary Code: "The Information Geometry of Primitive Accumulation: Quantifying Capitalist Enclosure in PIK2, Indonesia"


## Overview

This repository contains the data processing pipelines and analytical scripts required to reproduce the quantitative modeling and figures presented in the manuscript *"The Information Geometry of Primitive Accumulation: Quantifying Capitalist Enclosure in PIK2, Indonesia"*.

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

### Information Geometry on the Probability Simplex

To measure the "velocity" of landscape transformation, we calculate the Fisher-Rao geodesic distance on the probability simplex. For two class probability distributions $p$ and $q$, the distance is defined as:

$$d_{\mathrm{FR}}(p,q) = 2 \arccos\left(\sum_{i=1}^{N} \sqrt{p_i q_i}\right)$$

We complement this with Shannon entropy to track the pre-enclosure diversification phase:

$$H(p) = -\sum_{i=1}^{N} p_i \ln(p_i)$$

### Absorbing Markov Chains

Capitalist spatial accumulation is modeled as an absorbing Markov process, where *Built Area* represents the forced absorbing state ($P_{BB} = 1$). The fundamental matrix $\mathbf{N}$ is derived from the transient sub-matrix $\mathbf{Q}$:

$$\mathbf{N} = (\mathbf{I} - \mathbf{Q})^{-1}$$

The expected time to absorption (the "speed of enclosure") from any transient state $i$ is formulated as:

$$\mathbb{E}[\mathbf{T}] = \mathbf{N} \mathbf{1}$$

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

This project, including the code in this repository and the associated data hosted on OSF, is licensed under the MIT License. See the `LICENSE` file for details.
