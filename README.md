# SimDAT2D

A 2D X-ray pattern generator using PyFAI to simulate thin film data deposited onto single crystal substrates.

![Example Image](SimDat2D/sim.png)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
- [License](#license)

## Introduction

The SimDAT2D program can be used to generate synthetic X-ray scattering data from two or more distinct scattering contributions (e.g., isotropic, diffuse, and anisotropic). Each contribution is generated individually and combined as a linear combination into a single simulated 2D detector image. This allows for the relative intensities of each to be varied and for each signal to be independently known.

## Features

- Isotropic Scattering through Calibrants
- Anisotropic Scattering 
- Combination for Synthetic Thin Film 2D 
- Rotation and Integration
- Mask Creation

## Getting Started

Instructions on how to clone, and get the project running locally.

```bash
# Clone the repository
git clone https://github.com/dnalverson/SimDAT2D

# Change into the project directory
cd SimDAT2D

# Install dependencies
pip install -e .

# Start the project
import SimDAT2D as sd

```
## License

This project is licensed under the [BSD 3-Clause License](https://spdx.org/licenses/BSD-3-Clause-Clear.html) - see the [LICENSE](LICENSE.md) file for details.