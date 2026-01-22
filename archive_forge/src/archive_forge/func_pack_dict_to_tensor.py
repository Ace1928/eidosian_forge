import json
import shlex
import subprocess
from typing import Tuple
import torch
def pack_dict_to_tensor(source_dict):
    """
    Pack a dictionary into a torch tensor for storing quant_state items in state_dict.

    Parameters:
    - source_dict: The dictionary to be packed.

    Returns:
    A torch tensor containing the packed data.
    """
    json_str = json.dumps(source_dict)
    json_bytes = json_str.encode('utf-8')
    tensor_data = torch.tensor(list(json_bytes), dtype=torch.uint8)
    return tensor_data