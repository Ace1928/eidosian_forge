import gc
import random
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import top_k_top_p_filtering
from .import_utils import is_npu_available, is_xpu_available
def convert_to_scalar(stats: Dict) -> Dict:
    """
    Converts the stats from a flattened dict to single scalar dicts
    """
    tensorboard_stats = {}
    for k, v in stats.items():
        if (isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)) and (len(v.shape) == 0 or (len(v.shape) == 1 and v.shape[0] == 1)):
            v = v.item()
        tensorboard_stats[k] = v
    return tensorboard_stats