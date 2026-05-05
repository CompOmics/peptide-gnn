__version__ = "0.1.0"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'

import molgraph

from peptide_gnn import util