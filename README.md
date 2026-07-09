# PeptideGNN

A GNN-RNN hybrid model for predicting and explaining peptide LC-RT data.

## Overview

*To reproduce the results* of the paper, check out [Run it yourself](#run-it-yourself). 

*To build your own GNN-RNN hybrid model* on your own datasets, check out [Build it yourself](#build-it-yourself).

## Run it yourself

> [!NOTE]
> Results may very slightly differ from the paper due to the stochastic nature of neural nets.

Installation should take less than 30 minutes and is tested on a linux environment. Predictions should run within 30 minutes on normal hardware and much faster on a GPU.

### 1. Clone the repository

```bash
git clone git@github.com:CompOmics/peptide-gnn.git
cd peptide-gnn
```

### 2. Installation

We recommend using a virtual environment. Choose the installation that matches your hardware:

For GPU (CUDA) users:

```bash
pip install .[gpu]
```

For CPU users:

```bash
pip install .
```

### 3. Train and explain

Run the training pipeline using the provided sample data. By default, results are saved to `./output/`.

```bash
pepgnn run ./data/ --epochs 100
```

### 4. Visualize results

We provide a Jupyter notebook for post-hoc analysis and visualization:

1. Launch Jupyter: `jupyter notebook`
2. Navigate to `notebooks/vis.ipynb`
3. Run all cells to generate plots and model explanations.

## Build it yourself

If you are interested in building your own GNN-RNN hybrid model, check out [MolCraft](https://github.com/CompOmics/molcraft).

> [!NOTE]
> While MolCraft provides an improved API for molecular GNN-building, the underlying building blocks differ slightly from MolGraph, which may result in different outcomes.

