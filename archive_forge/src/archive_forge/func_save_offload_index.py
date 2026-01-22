import json
import os
from collections.abc import Mapping
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from safetensors import safe_open
def save_offload_index(index, offload_folder):
    if index is None or len(index) == 0:
        return
    offload_index_file = os.path.join(offload_folder, 'index.json')
    if os.path.isfile(offload_index_file):
        with open(offload_index_file, encoding='utf-8') as f:
            current_index = json.load(f)
    else:
        current_index = {}
    current_index.update(index)
    with open(offload_index_file, 'w', encoding='utf-8') as f:
        json.dump(current_index, f, indent=2)