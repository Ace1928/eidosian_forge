import torch.nn as nn
from .imports import is_fp8_available

    Returns whether a given model has some `transformer_engine` layer or not.
    