from tensorflow import keras 
from molgraph import layers 
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent.parent / 'data'

MODEL_CONFIG = {
    "gnn": {
        "num_layers": 3,
        "layer_type": layers.GATv2Conv,
        "layer_kwargs": {
            "units": 128,
            "normalization": None,
            "self_projection": True,
            "kernel_regularizer": keras.regularizers.L2(1e-5)
        },
        "trainable": True,
    },
    "rnn": {
        "num_layers": 1,
        "layer_type": keras.layers.LSTM,
        "layer_kwargs": {
            "units": 128,
        }
    },
    "dnn": {
        "num_layers": 2,
        "layer_type": keras.layers.Dense,
        "layer_kwargs": {
            "units": 1024,
            "activation": "relu",
            "kernel_regularizer": keras.regularizers.L2(1e-5),
        },
        "output_units": 1,
        "output_activation": "linear",
    }
}
