import json
import shlex
import subprocess
from typing import Tuple
import torch
def unpack_tensor_to_dict(tensor_data):
    """
    Unpack a torch tensor into a Python dictionary.

    Parameters:
    - tensor_data: The torch tensor containing the packed data.

    Returns:
    A Python dictionary containing the unpacked data.
    """
    json_bytes = bytes(tensor_data.cpu().numpy())
    json_str = json_bytes.decode('utf-8')
    unpacked_dict = json.loads(json_str)
    return unpacked_dict