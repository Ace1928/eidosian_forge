import json
import os
from collections.abc import Mapping
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from safetensors import safe_open
def load_offloaded_weight(weight_file, weight_info):
    shape = tuple(weight_info['shape'])
    if shape == ():
        shape = (1,)
    dtype = weight_info['dtype']
    if dtype == 'bfloat16':
        dtype = 'int16'
    weight = np.memmap(weight_file, dtype=dtype, shape=shape, mode='r')
    if len(weight_info['shape']) == 0:
        weight = weight[0]
    weight = torch.tensor(weight)
    if weight_info['dtype'] == 'bfloat16':
        weight = weight.view(torch.bfloat16)
    return weight