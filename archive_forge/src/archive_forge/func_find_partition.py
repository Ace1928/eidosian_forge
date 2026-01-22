from __future__ import annotations
import contextlib
import copy
import dataclasses
import datetime
import difflib
import enum
import functools
import io
import itertools
import os
import tempfile
import warnings
from typing import (
import numpy as np
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _experimental, _exporter_states, utils
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, onnx_proto_utils
from torch.types import Number
@_beartype.beartype
def find_partition(self, id: str) -> Optional['GraphInfo']:
    """Find the `GraphInfo` object with the given id."""
    if id == self.id:
        return self
    current_length = len(self.id)
    if len(id) > current_length:
        if id[current_length] == '0' and self.upper_graph_info is not None:
            return self.upper_graph_info.find_partition(id)
        elif id[current_length] == '1' and self.lower_graph_info is not None:
            return self.lower_graph_info.find_partition(id)
    return None