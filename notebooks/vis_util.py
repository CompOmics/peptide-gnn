
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from io import BytesIO
from typing import Union, List
import pickle
import ast
import pathlib
import molgraph
import math
import matplotlib.pyplot as plt
import pickle
import matplotlib.ticker as mtick
from matplotlib import font_manager

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
from rdkit.Chem import rdCoordGen
from rdkit.Geometry import Point3D
from PIL import Image
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from psm_utils import Peptidoform
from peptide_gnn.conf import DEFAULT_OUTPUT_DIR
from peptide_gnn.util import _DEFAULT_AMINO_ACIDS, compute_saliency, get_peptide_smiles, get_residue_indicator

molgraph.applications.proteomics.Peptide.register_residue_smiles(_DEFAULT_AMINO_ACIDS)


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

RENAME_DICT = {
    "LUNA_HILIC_fixed_mods":      "LUNA HILIC",
    "LUNA_SILICA_fixed_mods":     "LUNA SILICA",
    "SCX_fixed_mods":             "SCX",
    "Xbridge_fixed_mods":         "XBridge",
    "dia_fixed_mods":             "SWATH Library",
    "PXD005573_mcp" :             "DIA HF",
    "ATLANTIS_SILICA_fixed_mods": "ATLANTIS SILICA",
    "PXD029416_FA":               "FA",
    "PXD029416_AA":               "AcA",
    "PXD029416_FA_KO":            "FA-KU",
    "PXD029416_AA_KO":            "AcA-KU",
    "PXD029416_FA_MU":            "FA-UM",
    "PXD029416_AA_MU":            "AcA-UM",
    "PXD004919":                  "EG",
}

DATASET_ORDER = [
    "DIA HF", "SWATH Library",
    "FA", "FA-UM", "FA-KU",
    "AcA", "AcA-UM", "AcA-KU",
    "EG",
    "LUNA HILIC", "LUNA SILICA", "ATLANTIS SILICA",
    "XBridge", "SCX",
]

# Amino acids ordered by increasing polarity
AA_ORDERED = [
    "F", "L", "I", "W", "Y", "V", "P", "T",
    "C", "Q", "N", "M", "G", "A", "E", "D",
    "S", "R", "K", "H",
]

# ---------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------

def read_result(output_dir: str = None):
    """
    Reads all .csv and .pkl files from output folder into a single dataframe 
    """
    if not output_dir:
        output_dir = DEFAULT_OUTPUT_DIR

    dataframes = {}

    for file in pathlib.Path(output_dir).glob("*_prediction.csv"):
        name = file.stem.removesuffix("_prediction")

        saliency_path = file.parent / f"{name}_saliency.pkl"
        with open(saliency_path, "rb") as f:
            saliency_dict = pickle.load(f)

        prediction = pd.read_csv(file)
        prediction["saliency"] = prediction["peptide"].map(saliency_dict)

        dataframes[name] = prediction

    result = pd.concat(dataframes, axis=0)
    result = result.reset_index(level=0).rename(columns={"level_0": "dataset"})
    result = result.reset_index(drop=True)

    result["dataset"] = result["dataset"].map(lambda x: RENAME_DICT.get(x, x))
    return result

def prepare_datasets(df, selected_datasets=None, order=None):
    """
    Filter and order datasets in a single place.
    """
    df = df.copy()

    if selected_datasets:
        df = df[df["dataset"].isin(selected_datasets)]

    if order:
        order_index = {d: i for i, d in enumerate(order)}

        df = df.sort_values(
            by="dataset",
            key=lambda x: x.map(lambda d: order_index.get(d, len(order_index)))
        )
    return df

def normalize_saliency_dataset_wise(df, saliency_col="saliency"):
    """
    Adds a dataset-wise normalized saliency column to the DataFrame.

    Normalization is done per dataset using the maximum absolute value
    across all peptides in that dataset.

    Parameters
    ----------
    df : pd.DataFrame
    saliency_col : str
        Column with saliency values (list or string)
    """
    def to_array(x):
        if isinstance(x, str):
            return np.array(ast.literal_eval(x))
        return np.array(x)

    df[saliency_col] = df[saliency_col].apply(to_array)

    dataset_max = (
            df.dropna(subset=[saliency_col])
            .groupby("dataset")[saliency_col]
            .apply(lambda col: max(
                (np.max(np.abs(s)) for s in col if len(s) > 0),
                default=1.0
            ))
            .to_dict())
    def normalize(row):
        s = row[saliency_col]
        max_val = dataset_max.get(row["dataset"], 1.0)
        return s / max_val if max_val > 0 else s

    df["dataset_normalized_saliency"] = df.apply(normalize, axis=1)
    return df

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute regression metrics for a prediction DataFrame.

    Parameters
    ----------
    df : DataFrame with ``observed_rt`` and ``predicted_rt`` columns.

    Returns
    -------
    dict with keys: actual, predicted, mae, r_value, ci,
    error_percentile, min_rt, max_rt.
    """
    actual    = df["observed_rt"].astype(float)
    predicted = df["predicted_rt"].astype(float)
    errors    = predicted - actual
    abs_errors = np.abs(errors)

    return {
        "actual":           actual,
        "predicted":        predicted,
        "mae":              np.mean(abs_errors),
        "r_value":          np.corrcoef(actual, predicted)[0, 1],
        "ci":               np.percentile(abs_errors, 95) * 2,
        "error_percentile": np.percentile(abs_errors, 99),
        "min_rt":           actual.min(),
        "max_rt":           actual.max(),
    }

def plot_single_performance(ax, name: str, df: pd.DataFrame) -> None:
    """Plot a single observed vs predicted scatter on a given axis."""
    m = compute_metrics(df)
    actual, predicted = m["actual"], m["predicted"]

    ax.scatter(actual, predicted, s=5, alpha=0.8, color="black")
    ax.axline((0, 0), slope=1, color="grey", linewidth=0.8)

    line = np.linspace(m["min_rt"] - 100, m["max_rt"] + 100, 100)
    ep = m["error_percentile"]

    ax.plot(line, line + ep, "r--", linewidth=0.5)
    ax.plot(line, line - ep, "r--", linewidth=0.5)

    ax.set_title(
        f"{name}\nR={m['r_value']:.3f}, MAE={m['mae']:.3f}, Δt₉₅%={m['ci']:.2f}",
        fontsize=8
    )

    pad = 50 if name in ("SWATH Library", "DIA HF") else 10
    ax.set_xlim(m["min_rt"] - pad, m["max_rt"] + pad)
    ax.set_ylim(m["min_rt"] - pad, m["max_rt"] + pad)

    ax.tick_params(axis="both", labelsize=6, width=0.3, length=2, pad=0.5)
    ax.set_xlabel("Observed RT (min)", fontsize=7)
    ax.set_ylabel("Predicted RT (min)", fontsize=7)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(True)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)

def plot_performance(
    result: pd.DataFrame,
    dataset_col: str = "dataset",
    order: list[str] | None = None,
) -> None:
    if order is not None:
        grouped = [
            (name, result[result[dataset_col] == name])
            for name in order
            if name in result[dataset_col].values
        ]
    else:
        grouped = list(result.groupby(dataset_col))

    n = len(grouped)

    import math
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (name, df_subset) in zip(axes, grouped):
        plot_single_performance(ax, name, df_subset)

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Plotting amino acid contribution
# ---------------------------------------------------------------------------

def sum_to_aa_saliency_values(
    maps: list,
    peptides: list[str],
) -> list:
    """
    Aggregate atomic saliency values to residue-level by summing atoms
    belonging to each residue.

    Parameters
    ----------
    maps     : list of atomic saliency arrays (one per peptide).
    peptides : corresponding peptide sequences.

    Returns
    -------
    List of numpy arrays, one per peptide, with one value per residue.
    """
    result = []
    for peptide, atom_values in zip(peptides, maps):
        smiles = get_peptide_smiles(peptide)
        residue_indicator = get_residue_indicator(peptide)
        assert len(atom_values) == len(residue_indicator)

        map_t      = tf.convert_to_tensor(atom_values,       dtype=tf.float32)
        residue_t  = tf.convert_to_tensor(residue_indicator, dtype=tf.int32)
        segment    = tf.math.segment_sum(map_t, residue_t)
        result.append(segment.numpy())
    return result

def normalize_saliency_peptide_wise(
    df: pd.DataFrame,
    selected_datasets: list[str],
    saliency_col: str = "dataset_normalized_saliency",
    dataset_col: str = "dataset",
    peptide_col: str = "peptide",
) -> pd.DataFrame:
    """
    Apply peptide-wise normalisation across a pair (or more) of datasets.

    For each peptide the normalisation factor is the maximum absolute
    saliency value observed across *all* selected datasets, so that both
    panels of a comparison chart share a common colour scale.

    Parameters
    ----------
    df                : flat DataFrame with all datasets.
    selected_datasets : the two (or more) dataset names to compare.
    saliency_col      : column with per-atom saliency arrays.
    dataset_col       : column identifying the dataset.
    peptide_col       : column holding peptide sequences.

    Returns
    -------
    A filtered copy of *df* (only selected_datasets rows) with an extra
    column ``peptide_normalized_saliency``.
    """
    sub = df[df[dataset_col].isin(selected_datasets)].copy()

    # Peptide-level max-abs across all selected datasets
    peptide_max = (
        sub.groupby(peptide_col)[saliency_col]
        .apply(lambda col: max(
            (np.max(np.abs(np.array(s))) for s in col if len(np.array(s)) > 0),
            default=1.0
        ))
    )

    def _norm(row):
        mx = peptide_max.get(row[peptide_col], 1.0)
        return np.array(row[saliency_col]) / (mx if mx > 0 else 1.0)

    sub["peptide_normalized_saliency"] = sub.apply(_norm, axis=1)
    return sub

def build_aa_saliency_datasets(
    df: pd.DataFrame,
    saliency_col: str = "peptide_normalized_saliency",
    dataset_col: str = "dataset",
    peptide_col: str = "peptide",
) -> dict[str, dict[str, np.ndarray]]:
    """
    Convert per-atom saliency values to residue-level saliency by summing
    atoms belonging to each residue (TF segment sum).

    Parameters
    ----------
    df           : DataFrame with ``peptide_normalized_saliency`` column
                   (output of :func:`normalize_saliency_peptide_wise`).
    saliency_col : column with per-atom saliency arrays.
    dataset_col  : column identifying the dataset.
    peptide_col  : column holding peptide sequences.

    Returns
    -------
    dict: dataset name → {peptide: residue-level saliency array}
    """
    result = {}
    for name, group in df.groupby(dataset_col):
        peptide_dict = {}
        for _, row in group.iterrows():
            peptide   = row[peptide_col]
            atom_vals = np.array(row[saliency_col])
            residue_indicator = get_residue_indicator(peptide)
            if len(atom_vals) != len(residue_indicator):
                continue
            map_t   = tf.convert_to_tensor(atom_vals, dtype=tf.float32)
            res_t   = tf.convert_to_tensor(residue_indicator, dtype=tf.int32)
            peptide_dict[peptide] = tf.math.segment_sum(map_t, res_t).numpy()
        result[name] = peptide_dict
    return result

def convert_to_hashable(item):
    """
    Convert a parsed-sequence item (aa_letter, modification) into a
    hashable tuple suitable for use as a dict key.
    """
    key, value = item
    if isinstance(value, list):
        value = tuple(value)
    return (key, value)

def add_aa_saliency(
    df: pd.DataFrame,
    saliency_col: str = "dataset_normalized_saliency",
    peptide_col: str = "peptide",
    result_col: str = "amino_acid_saliency",
) -> pd.DataFrame:
    """
    Segment-sum atomic saliency values to residue level and store the result
    as a new column in *df*.

    This takes longer to compute (~3 min for large datasets). Run it once after
    loading data and reuse ``result_col`` for all subsequent plotting calls 
    where amino acid contributions are being visualized.

    Parameters
    ----------
    df           : flat DataFrame with per-atom saliency arrays.
    saliency_col : column with per-atom saliency arrays (input).
    peptide_col  : column with peptide sequences.
    result_col   : name of the new column to add (default "amino_acid_saliency").

    Returns
    -------
    The same DataFrame with *result_col* added in-place (also returned for
    convenience).

    Example
    -------
    result = add_aa_saliency(result)   # run once
    result.to_pickle("result_with_aa.pkl")  # optional: cache to disk
    """
    aa_saliency = []
    for _, row in df.iterrows():
        peptide   = row[peptide_col]
        atom_vals = np.array(row[saliency_col])
        residue_indicator = get_residue_indicator(peptide)
        if len(atom_vals) != len(residue_indicator):
            aa_saliency.append(None)
            continue
        map_t = tf.convert_to_tensor(atom_vals, dtype=tf.float32)
        res_t = tf.convert_to_tensor(residue_indicator, dtype=tf.int32)
        aa_saliency.append(tf.math.segment_sum(map_t, res_t).numpy().tolist())
    df[result_col] = aa_saliency
    return df

def compute_aa_mean_saliency(
    df: pd.DataFrame,
    aa_saliency_col: str = "amino_acid_saliency",
    dataset_col: str = "dataset",
    peptide_col: str = "peptide",
    aa_ordered: list[str] = AA_ORDERED,
) -> dict[str, dict]:
    """
    For each dataset compute per-amino-acid mean saliency from pre-computed
    residue-level saliency values.

    Requires ``add_aa_saliency()`` to have been called first so that
    *aa_saliency_col* exists in *df*.

    Parameters
    ----------
    df             : flat DataFrame with an ``amino_acid_saliency`` column.
    aa_saliency_col: column with residue-level saliency arrays (one value per
                     residue, produced by ``add_aa_saliency``).
    dataset_col    : column identifying the dataset.
    peptide_col    : column with peptide sequences.
    aa_ordered     : amino acid display order for sorting results.

    Returns
    -------
    dict: dataset name → {"letters": [...], "values": [...]}
    """
    if aa_saliency_col not in df.columns:
        raise ValueError(
            f"Column '{aa_saliency_col}' not found. "
            "Run add_aa_saliency(result) first to compute residue-level saliency."
        )

    results = {}
    aa_index = {aa: i for i, aa in enumerate(aa_ordered)}

    for name, group in df.groupby(dataset_col):
        value_dict: dict = defaultdict(list)
        aa_list = []

        for _, row in group.iterrows():
            aa_vals = row[aa_saliency_col]
            if aa_vals is None:
                continue
            aa_vals     = np.array(aa_vals)
            parsed_seq  = Peptidoform(row[peptide_col]).parsed_sequence
            if len(aa_vals) != len(parsed_seq):
                continue
            for val, parsed in zip(aa_vals, parsed_seq):
                letter = convert_to_hashable(parsed)
                value_dict[letter].append(float(val))
                if letter not in aa_list:
                    aa_list.append(letter)

        aa_list     = sorted(aa_list, key=lambda x: aa_index.get(x[0], float("inf")))
        mean_dict   = {letter: float(np.mean(vals)) for letter, vals in value_dict.items()}
        all_keys    = {convert_to_hashable(item): 0.0 for item in aa_list}
        mean_values = {**all_keys, **mean_dict}

        results[name] = {
            "letters": [
                k[0].lower()
                if k[1] is not None and "GenericModification" in str(k[1])
                else k[0]
                for k in mean_values
            ],
            "values": list(mean_values.values()),
        }
    return results



def compute_neighbor_heatmaps(
    df: pd.DataFrame,
    positions: list[int],
    directions: list[str],
    aa_saliency_col: str = "amino_acid_saliency",
    dataset_col: str = "dataset",
    peptide_col: str = "peptide",
    aa_ordered: list[str] = AA_ORDERED,
) -> dict[str, dict]:
    """
    Compute N-term / C-term neighbour heatmaps for each dataset in *df*.

    For each residue aa1, the heatmap value at (aa1, aa2) is the mean
    saliency of aa1 when aa2 appears at a given sequence distance away.
    Reads from the pre-computed ``amino_acid_saliency`` column — run
    ``add_aa_saliency()`` first.

    Parameters
    ----------
    df              : flat DataFrame with an ``amino_acid_saliency`` column.
    positions       : sequence distances to analyse, e.g. [1, 5, 10].
    directions      : ["N-term", "C-term"] or a subset.
    aa_saliency_col : column with residue-level saliency arrays.
    dataset_col     : column identifying the dataset.
    peptide_col     : column with peptide sequences.
    aa_ordered      : amino acid display order for rows/columns.

    Returns
    -------
    Nested dict: dataset → (direction, position) → {"mean": ndarray, "counts": ndarray}
    """
    if aa_saliency_col not in df.columns:
        raise ValueError(
            f"Column '{aa_saliency_col}' not found. "
            "Run add_aa_saliency(result) first."
        )

    all_results = {}

    for name, group in df.groupby(dataset_col):
        heatmaps = {}
        for direction in directions:
            for position in positions:
                neighbor_value    = defaultdict(lambda: defaultdict(float))
                count_occurrences = defaultdict(lambda: defaultdict(int))

                for _, row in group.iterrows():
                    aa_vals = row[aa_saliency_col]
                    if aa_vals is None:
                        continue
                    aa_vals    = np.array(aa_vals)
                    parsed_seq = Peptidoform(row[peptide_col]).parsed_sequence
                    if len(aa_vals) != len(parsed_seq):
                        continue

                    if direction == "N-term":
                        for i in range(position, len(parsed_seq)):
                            aa1 = parsed_seq[i][0]
                            aa2 = parsed_seq[i - position][0]
                            neighbor_value[aa1][aa2]    += float(aa_vals[i])
                            count_occurrences[aa1][aa2] += 1
                    else:  # C-term
                        for i in range(len(parsed_seq) - position):
                            aa1 = parsed_seq[i][0]
                            aa2 = parsed_seq[i + position][0]
                            neighbor_value[aa1][aa2]    += float(aa_vals[i])
                            count_occurrences[aa1][aa2] += 1

                n_aa           = len(aa_ordered)
                heatmap_sum    = np.zeros((n_aa, n_aa))
                heatmap_counts = np.zeros((n_aa, n_aa))
                for i, aa1 in enumerate(aa_ordered):
                    for j, aa2 in enumerate(aa_ordered):
                        heatmap_counts[i, j] = count_occurrences[aa1][aa2]
                        heatmap_sum[i, j]    = neighbor_value[aa1][aa2]

                heatmaps[(direction, position)] = {
                    "mean": np.divide(
                        heatmap_sum, heatmap_counts,
                        out=np.zeros_like(heatmap_sum),
                        where=heatmap_counts != 0,
                    ),
                    "counts": heatmap_counts,
                }
        all_results[name] = heatmaps
    return all_results

def plot_aa_bar_charts(
    df: pd.DataFrame,
    aa_saliency_col: str = "amino_acid_saliency",
    dataset_col: str = "dataset",
    peptide_col: str = "peptide",
    dataset_order: list[str] = None,
    n_cols: int = 5,
    set_type: str = None,
    set_col: str = "set_type",
    save_path: str | None = None,
) -> None:
    """
    Grid of bar charts showing mean amino acid saliency value per dataset.

    Positive mean saliency (green) means the residue increases predicted RT;
    negative (red) means it decreases RT. Amino acids are ordered by increasing polarity
    and labelled above/below each bar. Lower-case letter indicate modified amino acid.

    Parameters
    ----------
    df            : flat DataFrame with all datasets and saliency values.
    aa_saliency_col : column with residue-level saliency arrays produced by
                    ``add_aa_saliency()``. Must exist before calling this function.
    dataset_col   : column identifying the dataset.
    peptide_col   : column with peptide sequences.
    dataset_order : list controlling panel order; datasets not present are
                    skipped. Defaults to DATASET_ORDER.
    n_cols        : number of columns in the subplot grid (default 5).
    set_type      : which split to use — "test", "train", or None for all rows.
    set_col       : column name for the split indicator.
    save_path     : optional base path to save the figure, e.g. "output/aa_saliency".
                    Saves both <save_path>.pdf and <save_path>.tiff at 600 dpi.
    """
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1)
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "axes.grid":          False,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "svg.fonttype":       "none",
        "text.color":         "black",
        "axes.labelcolor":    "black",
        "axes.edgecolor":     "black",
        "xtick.color":        "black",
        "ytick.color":        "black",
        "axes.titlecolor":    "black",
        "axes.linewidth":     0.4,
    })

    if dataset_order is None:
        dataset_order = DATASET_ORDER

    plot_df = df[df[set_col] == set_type].copy() if set_type else df.copy()

    results = compute_aa_mean_saliency(
        plot_df,
        aa_saliency_col=aa_saliency_col,
        dataset_col=dataset_col,
        peptide_col=peptide_col,
    )

    ordered = [(name, results[name]) for name in dataset_order if name in results]
    if not ordered:
        ordered = list(results.items())

    mm_to_in = 1 / 25.4
    col_width = 180 * mm_to_in
    n_rows    = -(-len(ordered) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(col_width, n_rows * 0.4 * 88 * mm_to_in),
        constrained_layout=True,
    )
    axes = np.array(axes).flatten()

    for i, (name, data) in enumerate(ordered):
        ax      = axes[i]
        letters = data["letters"]
        values  = data["values"]
        colors  = ["#2ca02c" if v > 0 else "#d62728" for v in values]

        sns.barplot(
            x=letters, y=values,
            ax=ax,
            hue=letters,
            palette=colors,
            edgecolor=None,
            linewidth=0,
            errorbar=None,
            width=0.5,
        )

        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_title(name, fontsize=6, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(
            axis="y", labelsize=4, pad=0.5,
            width=0.4, length=2, direction="out",
        )

        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.4)

        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune=None))
        formatter = mtick.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(5)
        offset_text.set_ha("center")
        offset_text.set_va("bottom")
        offset_text.set_position((0, 0.03))

        local_max = max(abs(min(values)), abs(max(values)), 1e-12)
        ax.set_ylim(-local_max, local_max)

        offset = local_max * 0.05
        for x_pos, y_val, letter in zip(np.arange(len(values)), values, letters):
            ax.text(
                x_pos, y_val + np.sign(y_val) * offset,
                letter,
                ha="center",
                va="bottom" if y_val >= 0 else "top",
                fontsize=5,
            )


    for ax in axes[len(ordered):]:
        fig.delaxes(ax)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(f"{save_path}.pdf",  format="pdf", bbox_inches="tight")
        fig.savefig(f"{save_path}.tiff", format="tiff", dpi=600, bbox_inches="tight")

    plt.show()
    plt.close(fig)

def plot_neighbor_heatmaps(
    df: pd.DataFrame,
    aa_saliency_col: str = "amino_acid_saliency",
    dataset_col: str = "dataset",
    peptide_col: str = "peptide",
    positions: list[int] = None,
    directions: list[str] = None,
    set_type: str = None,
    set_col: str = "set_type",
) -> None:
    """
    For each dataset plot a grid of amino-acid neighbour heatmaps.

    Each cell (aa1, aa2) shows the mean saliency of aa1 when aa2 appears
    at the given sequence distance (N-terminal or C-terminal direction).
    All panels within one dataset share the same colour scale.
    Reads from the pre-computed ``amino_acid_saliency`` column — run
    ``add_aa_saliency()`` first.

    Parameters
    ----------
    df              : flat DataFrame with an ``amino_acid_saliency`` column.
    aa_saliency_col : column with residue-level saliency arrays produced by
                      ``add_aa_saliency()``.
    dataset_col     : column identifying the dataset.
    peptide_col     : column with peptide sequences.
    positions       : sequence distances to analyse, default [1, 5, 10].
    directions      : directions to analyse, default ["N-term", "C-term"].
    set_type        : which split to use, e.g. "test". Set to None for all.
    set_col         : column name for the split indicator.
    """
    if positions  is None: positions  = [1, 5, 10]
    if directions is None: directions = ["N-term", "C-term"]

    import warnings
    warnings.filterwarnings("ignore")

    cmap = LinearSegmentedColormap.from_list(
        "rwg", ["#cc1010", "white", "#0e870e"]
    )

    plot_df = df[df[set_col] == set_type].copy() if set_type else df.copy()

    all_results = compute_neighbor_heatmaps(
        plot_df, positions, directions,
        aa_saliency_col=aa_saliency_col,
        dataset_col=dataset_col,
        peptide_col=peptide_col,
    )

    plot_order = [(d, p) for p in positions for d in directions]

    for name, dataset_results in all_results.items():
        all_values = [v["mean"] for v in dataset_results.values()]
        max_abs    = max(np.max(np.abs(m)) for m in all_values)

        n_rows = len(positions)
        n_cols = len(directions)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 7))
        axes = np.array(axes).flatten()

        img = None
        for ax, key in zip(axes, plot_order):
            direction, position = key
            data = dataset_results[key]
            img  = sns.heatmap(
                data=data["mean"],
                ax=ax,
                yticklabels=AA_ORDERED,
                xticklabels=AA_ORDERED,
                annot=False,
                cmap=cmap,
                center=0,
                vmin=-max_abs,
                vmax=max_abs,
                cbar=False,
                linewidths=0.5,
            )
            ax.set_title(f"{direction} — position {position}", fontsize=8)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.tick_params(axis="both", labelsize=6)

        cbar_ax = fig.add_axes([0.99, 0.025, 0.02, 0.2])
        fig.colorbar(img.collections[0], cax=cbar_ax)
        cbar_ax.tick_params(labelsize=6)

        plt.suptitle(name, fontsize=10)
        plt.tight_layout()
        plt.show()

def plot_comparison_chart(
    df: pd.DataFrame,
    left: str,
    right: str,
    saliency_col: str = "dataset_normalized_saliency",
    dataset_col: str = "dataset",
    peptide_col: str = "peptide",
    max_peptides: int = 30,
    set_type: str = "test",
    set_col: str = "set_type",
    cell_size: float = 0.22,
    annot_size: float = 7,
    title_size: float = 11,
) -> None:

    cmap = LinearSegmentedColormap.from_list(
        "rwg", ["#cc1010", "white", "#0e870e"]
    )
    cmap.set_bad("white")

    # Plot peptides without plotting PTMs
    plot_df = df.copy()
    plot_df["_clean_peptide"] = plot_df[peptide_col].apply(
        lambda p: Peptidoform(p).sequence
    )

    # Filter rows
    plot_df = plot_df[plot_df[set_col] == set_type] if set_type else plot_df
    plot_df = plot_df[plot_df[dataset_col].isin([left, right])]

    if plot_df[dataset_col].nunique() < 2:
        missing = {left, right} - set(plot_df[dataset_col].unique())
        print(f"Dataset(s) not found in df: {missing}")
        return


    common_peptides = get_common_peptides(
        plot_df,
        datasets=[left, right],
        dataset_col=dataset_col,
        peptide_col="_clean_peptide",
    )

    if not common_peptides:
        print("No peptides are common between the two selected datasets.")
        return

    common_sorted = sorted(
        common_peptides,
        key=lambda p: Peptidoform(p).theoretical_mass,
    )[:max_peptides]

    plot_df = plot_df[plot_df["_clean_peptide"].isin(common_sorted)]

 
    peptide_max = (
        plot_df.groupby("_clean_peptide")[saliency_col]
        .apply(lambda col: max(
            (np.max(np.abs(np.array(s))) for s in col if len(np.array(s)) > 0),
            default=1.0,
        ))
    )

    def _norm_and_sum(row):
        atom_vals = np.array(row[saliency_col])
        mx = peptide_max.get(row["_clean_peptide"], 1.0)
        normed = atom_vals / (mx if mx > 0 else 1.0)

        residue_indicator = get_residue_indicator(row["_clean_peptide"])

        if len(normed) != len(residue_indicator):
            return None

        map_t = tf.convert_to_tensor(normed, dtype=tf.float32)
        res_t = tf.convert_to_tensor(residue_indicator, dtype=tf.int32)

        return tf.math.segment_sum(map_t, res_t).numpy().tolist()

    plot_df = plot_df.copy()
    plot_df["_aa_norm"] = plot_df.apply(_norm_and_sum, axis=1)
    plot_df = plot_df[plot_df["_aa_norm"].notna()]


    def _peptide_dict(dataset_name):
        sub = plot_df[plot_df[dataset_col] == dataset_name]
        return {
            row["_clean_peptide"]: np.array(row["_aa_norm"])
            for _, row in sub.iterrows()
        }

    left_dict = _peptide_dict(left)
    right_dict = _peptide_dict(right)

    common_final = [p for p in common_sorted
                    if p in left_dict and p in right_dict]

    if not common_final:
        print("No common peptides remained after normalisation.")
        return

    max_len = max(len(left_dict[p]) for p in common_final)

    def _pad(peptide_list, source_dict):
        return np.array([
            list(source_dict[p]) + [np.nan] * (max_len - len(source_dict[p]))
            for p in peptide_list
        ])

    left_data = _pad(common_final, left_dict)
    right_data = _pad(common_final, right_dict)

 
    left_labels = np.array([
        list(p[:len(left_dict[p])]) + [""] * (max_len - len(left_dict[p]))
        for p in common_final
    ], dtype=object)

    left_labels = np.stack(left_labels)
    right_labels = left_labels.copy()


    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "axes.linewidth": 0.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    n = len(common_final)
    fig_w = 2 * max_len * cell_size
    fig_h = n * cell_size

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, wspace=0.1)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

 
    vmin = float(np.nanmin(np.concatenate([
        left_data.flatten(), right_data.flatten()
    ])))
    vmax = float(np.nanmax(np.concatenate([
        left_data.flatten(), right_data.flatten()
    ])))

    max_abs = max(abs(vmin), abs(vmax))

    hm_kw = dict(
        cmap=cmap,
        vmin=-max_abs,
        vmax=max_abs,
        fmt="s",
        xticklabels=False,
        yticklabels=False,
        annot_kws={"size": annot_size, "family": "Arial"},
        linewidths=0.3,
        cbar=False,
        square=True,
    )

    mask_left = np.isnan(left_data)
    mask_right = np.isnan(right_data)

    sns.heatmap(left_data, annot=left_labels, ax=axes[0],
                mask=mask_left, **hm_kw)

    axes[0].invert_xaxis()

    sns.heatmap(right_data, annot=right_labels, ax=axes[1],
                mask=mask_right, **hm_kw)

  
    for spine in axes[0].spines.values():
        spine.set_linewidth(0.4)

    axes[1].spines["left"].set_visible(False)

    for spine in ["top", "right", "bottom"]:
        axes[1].spines[spine].set_linewidth(0.4)

    fig.text(0.35, 0.88, left, fontsize=title_size, ha="center",
             va="bottom", fontfamily="Arial", transform=fig.transFigure)

    fig.text(0.68, 0.88, right, fontsize=title_size, ha="center",
             va="bottom", fontfamily="Arial", transform=fig.transFigure)


    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.018])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=-max_abs, vmax=max_abs),
            cmap=cmap,
        ),
        cax=cbar_ax,
        orientation="horizontal",
    )

    cb.ax.tick_params(labelsize=7, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)

    plt.show()

# ---------------------------------------------------------------------------
# Molecule drawings in RDKit
# ---------------------------------------------------------------------------

def get_common_peptides(
    df,
    datasets: list[str] | None = None,
    dataset_col: str = "dataset",
    peptide_col: str = "peptide",
) -> set:
    """
    Return peptides present in every dataset in *datasets*.

    Parameters
    ----------
    df       : flat DataFrame with all data.
    datasets : list of dataset names to intersect. When None, uses all
               datasets present in *df*.
    """
    if datasets is not None:
        df = df[df[dataset_col].isin(datasets)]

    n_datasets = df[dataset_col].nunique()

    return set(
        df.groupby(peptide_col)[dataset_col]
          .nunique()
          .loc[lambda x: x == n_datasets]
          .index
    )

def merge_molecules_with_offset(
    molecule_list: List[Union[str, Chem.Mol]],
    offset: float = 3.0,
) -> Chem.Mol:
    """
    Merge multiple RDKit molecules into one object, separating them
    horizontally by *offset* units.
    """
    merged = Chem.RWMol()
    current_offset = 0.0
    atom_map = {}

    for mol in molecule_list:
        if not isinstance(mol, Chem.Mol):
            mol = Chem.MolFromSmiles(mol)

        rdCoordGen.AddCoords(mol)
        conf = mol.GetConformer()

        if not merged.GetNumConformers():
            merged.AddConformer(Chem.Conformer())

        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            new_pos = Point3D(pos.x + current_offset, pos.y, pos.z)
            idx = merged.AddAtom(atom)
            merged.GetConformer().SetAtomPosition(idx, new_pos)
            atom_map[atom.GetIdx()] = idx

        for bond in mol.GetBonds():
            a1 = atom_map.get(bond.GetBeginAtomIdx())
            a2 = atom_map.get(bond.GetEndAtomIdx())
            if a1 is not None and a2 is not None:
                merged.AddBond(a1, a2, bond.GetBondType())

        x_coords = conf.GetPositions()[:, 0]
        current_offset += offset + x_coords.max() - x_coords.min()
        atom_map.clear()

    try:
        Chem.SanitizeMol(merged)
    except Exception as e:
        print(f"Sanitization warning: {e}")

    return merged.GetMol()

def draw_similarity_map(
    mol: Chem.Mol,
    weights: list,
    size: tuple[int, int],
    scale_factor: int = 4, 
    padding: float = 0.02,
    scale: float = 0.0,
) -> Image.Image:
    """
    Render a saliency / similarity map for *mol* and return a PIL Image.

    Renders at higher resolution and downsamples for sharpness.
    """

    if not mol.GetNumConformers():
        rdCoordGen.AddCoords(mol)

    high_res = (size[0] * scale_factor, size[1] * scale_factor)
    drawer = rdMolDraw2D.MolDraw2DCairo(*high_res)

    opts = drawer.drawOptions()
    opts.padding = padding

    opts.clearBackground = False
    opts.bondLineWidth = 1.5 * scale_factor
    opts.atomLabelFontSize = int(12 * scale_factor)


    SimilarityMaps.GetSimilarityMapFromWeights(
        mol=mol,
        weights=weights,
        scale=scale,
        draw2d=drawer,
    )

    drawer.FinishDrawing()

    img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")

    return img

def plot_qualitative_maps(
     molecules: list,
    maps_list: list,
    save_path: str,
    labels: list[str] | None = None,
    dpi: int = 600,
    fig_width_mm: float = 240,
    panel_height_mm: float = 60,
    padding: float = 0.02,
    scale: float = 0.0,
    n_cols: int | None = None,
    font_size_pt: int = 70,
    gap_px: int = 10,
):
    """High-quality qualitative saliency maps."""

    import math

    n = len(molecules)

   
    if n_cols is None:
        if n <= 3:
            n_cols = n
        elif n <= 6:
            n_cols = 3
        else:
            n_cols = math.ceil(math.sqrt(n))

    n_rows = math.ceil(n / n_cols)


    mm_to_inch = 1 / 25.4
    fig_w_px = int((fig_width_mm * mm_to_inch) * dpi)
    fig_h_px = int((panel_height_mm * n_rows * mm_to_inch) * dpi)

   
    total_hgap = gap_px * (n_cols - 1)
    total_vgap = gap_px * (n_rows - 1)

    panel_w = (fig_w_px - total_hgap) // n_cols
    panel_h = (fig_h_px - total_vgap) // n_rows

    
    
    images = []
    for mol, w in zip(molecules, maps_list):
        if not mol.GetNumConformers():
            Chem.rdCoordGen.AddCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DCairo(panel_w, panel_h)
        opts = drawer.drawOptions()
        opts.padding = min(padding, 0.02)

      
        SimilarityMaps.GetStandardizedWeights(w)

        SimilarityMaps.GetSimilarityMapFromWeights(
            mol=mol,
            weights=[float(x) for x in w],
            scale=scale,
            draw2d=drawer,
        )

        drawer.FinishDrawing()
        img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")
        images.append(img)

    
    combined = Image.new("RGBA", (fig_w_px, fig_h_px), (255, 255, 255, 255))
    draw = ImageDraw.Draw(combined)

    font_path = font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, font_size_pt)


    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols

        x = col * (panel_w + gap_px)
        y = row * (panel_h + gap_px)

        combined.paste(img, (x, y), img)

     
        if labels:
            label = labels[i]
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]

            x_center = x + panel_w // 2 - text_w // 2
            y_text = y + int(0.02 * panel_h)

            draw.text((x_center, y_text), label, fill="black", font=font)


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.save(save_path, format="PNG", dpi=(dpi, dpi))

def plot_quantitative_maps(
    molecules: list,
    maps_list: list,
    labels: list[str],
    save_path: str,
    dpi: int = 600,
    fig_width_mm: float = 240,
    panel_height_mm: float = 60,
    padding: float = 0.02,
    scale: float = 0.0,
    offset: float = 3.0,
    font_size_pt: int = 50,
):
    """
    High-quality quantitative saliency map (shared color scale). 
    All selected molecules are normalized as a one object in RDKit.
   
    """
    n = len(molecules)
    if n == 0:
        return

    mm_to_inch = 1 / 25.4
    fig_w_px = int((fig_width_mm * mm_to_inch) * dpi)
    fig_h_px = int((panel_height_mm * mm_to_inch) * dpi)

  
    merged = merge_molecules_with_offset(molecules, offset)
    merged_maps = np.concatenate(maps_list)


    drawer = rdMolDraw2D.MolDraw2DCairo(fig_w_px, fig_h_px)
    opts = drawer.drawOptions()
    opts.padding = min(padding, 0.02)

    SimilarityMaps.GetSimilarityMapFromWeights(
            mol=merged,
            weights=[float(w) for w in merged_maps],
            scale=scale,
            draw2d=drawer,
        )

    drawer.FinishDrawing()
    img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")

    # --- labels ---
    draw = ImageDraw.Draw(img)

    font_path = font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, font_size_pt)

    # evenly distribute labels across width
    block_w = fig_w_px / n

    for i, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]

        x_center = int(i * block_w + block_w / 2 - text_w / 2)
        y_text = int(0.05 * fig_h_px)

        draw.text((x_center, y_text), label, fill="black", font=font)

    # --- save ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path, format="PNG", dpi=(dpi, dpi))

def plot_peptide_maps(
    df_qual,
    df_quant,
    peptides,
    output_folder,
    max_peptides: int | None = None,
    random_seed: int | None = 42,
):
    """
    Save qualitative and quantitative saliency map images for each peptide.

    Only datasets that actually contain the peptide are included. Because
    *peptides* should already be the result of get_common_peptides() filtered
    to the same datasets present in df_qual/df_quant, every peptide should
    appear in every dataset.

    Parameters
    ----------
    df_qual       : split-filtered DataFrame; uses 'saliency' column.
    df_quant      : split-filtered DataFrame; uses 'dataset_normalized_saliency' column.
    peptides      : iterable of peptide sequences to render.
    output_folder : directory where PNG files are written.
    max_peptides  : maximum number of peptides to render. None renders all.
    random_seed   : seed for reproducible random sampling when max_peptides is
                    set. Use None for a different random selection each run.
    """
    datasets_qual  = list(df_qual["dataset"].unique())
    datasets_quant = list(df_quant["dataset"].unique())

    peptides = list(peptides)
    if max_peptides is not None and max_peptides < len(peptides):
        rng = np.random.default_rng(random_seed)
        peptides = list(rng.choice(peptides, size=max_peptides, replace=False))
    for peptide in peptides:
        smiles = molgraph.applications.proteomics.Peptide(peptide).smiles

        mols_q, maps_q, labels_q     = [], [], []
        mols_qt, maps_qt, labels_qt  = [], [], []

        # --- QUALITATIVE ---
        for dataset_name in datasets_qual:
            row = df_qual[
                (df_qual["dataset"] == dataset_name) &
                (df_qual["peptide"] == peptide)
            ]
            if row.empty:
                continue
            sal = row.iloc[0]["saliency"]
            if sal is None:
                continue
            mol = Chem.MolFromSmiles(smiles)
            rdCoordGen.AddCoords(mol)
            mols_q.append(mol)
            maps_q.append(np.array(sal))
            labels_q.append(dataset_name)

        # --- QUANTITATIVE ---
        for dataset_name in datasets_quant:
            row = df_quant[
                (df_quant["dataset"] == dataset_name) &
                (df_quant["peptide"] == peptide)
            ]
            if row.empty:
                continue
            sal = row.iloc[0]["dataset_normalized_saliency"]
            if sal is None:
                continue
            mol = Chem.MolFromSmiles(smiles)
            rdCoordGen.AddCoords(mol)
            mols_qt.append(mol)
            maps_qt.append(np.array(sal))
            labels_qt.append(dataset_name)

        # --- plotting ---
        if len(mols_qt) >= 2:
            plot_quantitative_maps(
                mols_qt,
                maps_qt,
                labels_qt,
                os.path.join(output_folder, f"{peptide}_quantitative.png"),
            )

        if mols_q:
            plot_qualitative_maps(
                mols_q,
                maps_q,
                save_path=os.path.join(output_folder, f"{peptide}_qualitative.png"),
                labels=labels_q,
            )

