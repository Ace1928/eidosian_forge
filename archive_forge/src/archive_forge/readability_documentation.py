from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Union
import torch
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics
Restore parameter and buffer names from original module.

        For each `get_attr` node, if the target is a str representing a parameter or buffer
        under `self.module`, we rename the parameter or buffer to its original name.
        The parameters and buffers between `self.module` and `self.original_nn_module` refer
        to the same objects, allowing us to use it as key to retrieve the original name.
        