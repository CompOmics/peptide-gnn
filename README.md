# PeptideGNN

A GNN-RNN hybrid model for predicting and explaining peptide LC-RT data.

## Overview

*To reproduce the results* of the paper, check out [Run it yourself](#run-it-yourself). 

*To build your own GNN-RNN hybrid model* on your own datasets, check out [Build it yourself](#build-it-yourself).

## Run it yourself

> [!IMPORTANT]
> The results may differ from the paper due to the stochastic nature of neural nets.

> [!NOTE]
> The steps below have only been run and tested in a linux environment.
> Training and explaining _a_ model on _a_ dataset should take roughly 10-60 minutes, depending on dataset size and hardware.

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

### 5. \[Optional\] Save and load a model

To save model(s) to disk, run `pepgnn` with the `--save-model` flag.

```bash
pepgnn run ./data/dia_fixed_mods.csv --epochs 100 --save-model
```

The saved model(s) can then be loaded and used in a (jupyter) notebook.

```python
from molcraft import applications
from tensorflow import keras
from peptide_gnn import util

featurizer = util.create_featurizer()
model = keras.models.load_model('./output/models/dia_fixed_mods_model.keras')
saliency = applications.proteomics.PeptideSaliency(model)

sequences, _ = util.load_dataset('./data/dia_fixed_mods.csv')
graphs = featurizer(sequences[:100]) # featurize the first 100 to reduce run time

predictions = model.predict(graphs)
saliencies = saliency(graphs.separate()).numpy().tolist()
```

## Build it yourself

If you are interested in building and deploying your own GNN-RNN hybrid model, check out [MolCraft](https://github.com/CompOmics/molcraft#hybrid-model-for-peptides).

> [!NOTE]
> While MolCraft provides an improved API for molecular GNN-building, the underlying building blocks differ slightly from MolGraph, which may result in different outcomes.

