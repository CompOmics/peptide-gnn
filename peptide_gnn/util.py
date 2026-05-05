import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import keras
import math

from psm_utils.io.peptide_record import PeptideRecordReader

from molgraph import layers
from molgraph import chemistry
from molgraph import applications

from peptide_gnn.conf import MODEL_CONFIG


def load_dataset(file_path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    peptides, rt = [], []
    for psm in PeptideRecordReader(str(file_path)):
        peptides.append(psm.peptidoform.proforma)
        rt.append(psm.retention_time)
    return np.array(peptides), np.array(rt, dtype=np.float32)

def shuffle_dataset(*dataset, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset[0]))
    return tuple(arr[indices] for arr in dataset)

def split_dataset(
    *dataset, 
    train_frac: float = 0.7,
    validation_frac: float = 0.15,
    test_frac: float = 0.15, 
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    assert train_frac + validation_frac + test_frac == 1.0
    n = len(dataset[0])
    train_end = int(train_frac * n)
    val_end = train_end + int(validation_frac * n)
    result = []
    for arr in dataset:
        result.append(
            (arr[:train_end], arr[train_end:val_end], arr[val_end:])
        )
    return tuple(result)

def write_records(
    x: np.ndarray,
    y: np.ndarray,
    encoder: applications.proteomics.PeptideGraphEncoder,
    path: str | pathlib.Path,
) -> None:
    path = pathlib.Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    with chemistry.tf_records.writer(
        path, num_files=math.ceil(len(x) / 1000) # 1000 examples per file
    ) as writer:
        writer.write(
            data={'x': x, 'y': y}, 
            encoder=encoder,
        )

def read_records(
    path: str | pathlib.Path,
    batch_size: int = 16,
    shuffle: bool = False,
) -> tf.data.Dataset:
    ds = chemistry.tf_records.load(
        path, 
        extract_tuple=('x', 'y'),
        shuffle_tf_records=shuffle
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    return ds

def create_atom_featurizer() -> chemistry.Featurizer:
    return chemistry.Featurizer([
        chemistry.features.Symbol({'C', 'N', 'P', 'H', 'S', 'O'}),
        chemistry.features.Hybridization(),
        chemistry.features.FormalCharge(),
        chemistry.features.TotalNumHs(),
        chemistry.features.Aromatic(),
        chemistry.features.CIPCode(),
        chemistry.features.ChiralCenter(),
        chemistry.features.TotalValence(),
        chemistry.features.NumRadicalElectrons(),
        chemistry.features.Degree(),
        chemistry.features.Hetero(),
        chemistry.features.HydrogenDonor(),
        chemistry.features.HydrogenAcceptor(),
        chemistry.features.RingSize(),
        chemistry.features.Ring(),
        chemistry.features.CrippenLogPContribution(),
        chemistry.features.CrippenMolarRefractivityContribution(),
        chemistry.features.TPSAContribution(),
        chemistry.features.LabuteASAContribution(),
        chemistry.features.GasteigerCharge(),
    ])

def create_bond_featurizer() -> chemistry.Featurizer:
    return chemistry.Featurizer([
        chemistry.features.BondType(),
        chemistry.features.Conjugated(),
        chemistry.features.Stereo(),
        chemistry.features.Rotatable(),
    ])

def create_featurizer() -> applications.proteomics.PeptideGraphEncoder:
    atom_featurizer = create_atom_featurizer()
    bond_featurizer = create_bond_featurizer()
    applications.proteomics.Peptide.register_residue_smiles(_DEFAULT_AMINO_ACIDS)
    return applications.proteomics.PeptideGraphEncoder(
        atom_featurizer, bond_featurizer, super_nodes=False
    )

def create_and_adapt_preprocessing_layers(
    ds: tf.data.Dataset,
) -> tuple[layers.MinMaxScaling, layers.MinMaxScaling]:
    node_preprocessing = layers.MinMaxScaling(
        feature='node_feature', feature_range=(0, 1), threshold=True
    )
    edge_preprocessing = layers.MinMaxScaling(
        feature='edge_feature', feature_range=(0, 1), threshold=True
    )
    node_preprocessing.adapt(ds.map(lambda x, y: x))
    edge_preprocessing.adapt(ds.map(lambda x, y: x))
    return node_preprocessing, edge_preprocessing

def create_model(
    ds: tf.data.Dataset,
) -> keras.Sequential:
    node_preprocessing, edge_preprocessing = create_and_adapt_preprocessing_layers(ds)
    return applications.proteomics.PeptideModel(
        config=MODEL_CONFIG,
        preprocessing=[node_preprocessing, edge_preprocessing],
        spec=ds.element_spec[0],
    )

def create_optimizer(
    start_lr: float = 1e-4,
    end_lr: float = 1e-6,
    decay_steps: int = None,
) -> tf.keras.optimizers.Adam:
    assert decay_steps is not None
    schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=start_lr,
        decay_steps=decay_steps,
        end_learning_rate=end_lr,
        power=1.0,
        cycle=False,
    )
    return tf.keras.optimizers.Adam(schedule, clipnorm=1.0)

def compute_saliency(model: keras.Model, dataset: tf.data.Dataset) -> None:
    saliency = applications.proteomics.PeptideSaliency(model)
    
    def atom_saliency(x) -> tuple[tf.RaggedTensor, tf.RaggedTensor]:
        return saliency(x.separate())

    saliency_maps = []
    for x_batch, _ in dataset:
        saliency_maps.extend(atom_saliency(x_batch).numpy().tolist())

    return saliency_maps

def get_peptide_smiles(sequence: str) -> str:
    return applications.proteomics.Peptide(sequence).smiles 

def get_residue_indicator(sequence: str) -> np.ndarray:
    peptide = applications.proteomics.Peptide(sequence)
    sizes = peptide.residue_sizes
    return np.repeat(np.arange(len(sizes)), sizes)


_DEFAULT_AMINO_ACIDS = {
    "A": "N[C@@H](C)C(=O)O",
    "C": "N[C@@H](CS)C(=O)O",
    "C[Carbamidomethyl]": "N[C@@H](CSCC(=O)N)C(=O)O",
    "D": "N[C@@H](CC(=O)O)C(=O)O",
    "E": "N[C@@H](CCC(=O)O)C(=O)O",
    "F": "N[C@@H](Cc1ccccc1)C(=O)O",
    "G": "NCC(=O)O",
    "H": "N[C@@H](CC1=CN=C-N1)C(=O)O",
    "I": "N[C@@H](C(CC)C)C(=O)O",
    "K": "N[C@@H](CCCCN)C(=O)O",
    "K[Acetyl]": "N[C@@H](CCCCNC(=O)C)C(=O)O",
    "K[Crotonyl]": "N[C@@H](CCCCNC(C=CC)=O)C(=O)O",
    "K[Dimethyl]": "N[C@@H](CCCCN(C)C)C(=O)O",
    "K[Formyl]": "N[C@@H](CCCCNC=O)C(=O)O",
    "K[Malonyl]": "N[C@@H](CCCCNC(=O)CC(O)=O)C(=O)O",
    "K[Methyl]": "N[C@@H](CCCCNC)C(=O)O",
    "K[Propionyl]": "N[C@@H](CCCCNC(=O)CC)C(=O)O",
    "K[Succinyl]": "N[C@@H](CCCCNC(CCC(O)=O)=O)C(=O)O",
    "K[Trimethyl]": "N[C@@H](CCCC[N+](C)(C)C)C(=O)O",
    "L": "N[C@@H](CC(C)C)C(=O)O",
    "M": "N[C@@H](CCSC)C(=O)O",
    "M[Oxidation]": "N[C@@H](CCS(=O)C)C(=O)O",
    "N": "N[C@@H](CC(=O)N)C(=O)O",
    "P": "N1[C@@H](CCC1)C(=O)O",
    "P[Oxidation]": "N1CC(O)C[C@H]1C(=O)O",
    "Q": "N[C@@H](CCC(=O)N)C(=O)O",
    "R": "N[C@@H](CCCNC(=N)N)C(=O)O",
    "R[Deamidated]": "N[C@@H](CCCNC(N)=O)C(=O)O",
    "R[Dimethyl]": "N[C@@H](CCCNC(N(C)C)=N)C(=O)O",
    "R[Methyl]": "N[C@@H](CCCNC(=N)NC)C(=O)O",
    "S": "N[C@@H](CO)C(=O)O",
    "T": "N[C@@H](C(O)C)C(=O)O",
    "V": "N[C@@H](C(C)C)C(=O)O",
    "W": "N[C@@H](CC(=CN2)C1=C2C=CC=C1)C(=O)O",
    "Y": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "Y[Nitro]": "N[C@@H](Cc1ccc(O)c(N(=O)=O)c1)C(=O)O",
    "Y[Phospho]": "N[C@@H](Cc1ccc(OP(O)(=O)O)cc1)C(=O)O",
    "[Acetyl]-A": "N(C(C)=O)[C@@H](C)C(=O)O",
    "[Acetyl]-C": "N(C(C)=O)[C@@H](CS)C(=O)O",
    "[Acetyl]-D": "N(C(=O)C)[C@H](C(=O)O)CC(=O)O",
    "[Acetyl]-E": "N(C(=O)C)[C@@H](CCC(O)=O)C(=O)O",
    "[Acetyl]-F": "N(C(C)=O)[C@@H](Cc1ccccc1)C(=O)O",
    "[Acetyl]-G": "N(C(=O)C)CC(=O)O",
    "[Acetyl]-H": "N(C(=O)C)[C@@H](Cc1[nH]cnc1)C(=O)O",
    "[Acetyl]-I": "N(C(=O)C)[C@@H]([C@H](CC)C)C(=O)O",
    "[Acetyl]-K": "N(C(C)=O)[C@@H](CCCCN)C(=O)O",
    "[Acetyl]-L": "N(C(=O)C)[C@@H](CC(C)C)C(=O)O",
    "[Acetyl]-M": "N(C(=O)C)[C@@H](CCSC)C(=O)O",
    "[Acetyl]-N": "N(C(C)=O)[C@@H](CC(=O)N)C(=O)O",
    "[Acetyl]-P": "N1(C(=O)C)CCC[C@H]1C(=O)O",
    "[Acetyl]-Q": "N(C(=O)C)[C@@H](CCC(=O)N)C(=O)O",
    "[Acetyl]-R": "N(C(C)=O)[C@@H](CCCN=C(N)N)C(=O)O",
    "[Acetyl]-S": "N(C(C)=O)[C@@H](CO)C(=O)O",
    "[Acetyl]-T": "N(C(=O)C)[C@@H]([C@H](O)C)C(=O)O",
    "[Acetyl]-V": "N(C(=O)C)[C@@H](C(C)C)C(=O)O",
    "[Acetyl]-W": "N(C(C)=O)[C@@H](Cc1c2ccccc2[nH]c1)C(=O)O",
    "[Acetyl]-Y": "N(C(C)=O)[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "[Acetyl]-M[Oxidation]": "N(C(=O)C)[C@@H](CCS(C)=O)C(=O)O",
    "[Acetyl]-S[Phospho]":"N(C(C)=O)C(C(O)=O)COP(O)(=O)O",
    "S[Phospho]": "N[C@@H](COP(O)(=O)O)C(=O)O",
    "N[Deamidated]": "N[C@@H](CC(O)=O)C(=O)O",
    "Q[Deamidated]":   "N[C@@H](CCC(=O)O)C(=O)O",
    "T[Phospho]": "N[C@H](C(=O)O)[C@@H](C)OP(O)(=O)O",
    "[Acetyl]-C[Carbamidomethyl]": "N(C(=O)C)C(CSCC(=O)N)C(=O)O",
    "[Acetyl]-T[Phospho]":"N(C(C)=O)C(C(C)OP(=O)(O)O)C(=O)O"
}
