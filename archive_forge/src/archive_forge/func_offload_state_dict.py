import json
import os
from collections.abc import Mapping
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from safetensors import safe_open
def offload_state_dict(save_dir: Union[str, os.PathLike], state_dict: Dict[str, torch.Tensor]):
    """
    Offload a state dict in a given folder.

    Args:
        save_dir (`str` or `os.PathLike`):
            The directory in which to offload the state dict.
        state_dict (`Dict[str, torch.Tensor]`):
            The dictionary of tensors to offload.
    """
    os.makedirs(save_dir, exist_ok=True)
    index = {}
    for name, parameter in state_dict.items():
        index = offload_weight(parameter, name, save_dir, index=index)
    save_offload_index(index, save_dir)