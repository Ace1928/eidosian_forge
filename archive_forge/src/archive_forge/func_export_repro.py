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
def export_repro(self, repro_dir: Optional[str]=None, name: Optional[str]=None) -> str:
    """Export the subgraph to ONNX along with the input/output data for repro.

        The repro directory will contain the following files::

            dir
            ├── test_<name>
            │   ├── model.onnx
            │   └── test_data_set_0
            │       ├── input_0.pb
            │       ├── input_1.pb
            │       ├── output_0.pb
            │       └── output_1.pb

        Args:
            repro_dir: The directory to export the repro files to. Defaults to current
                working directory if None.
            name: An optional name for the test case folder: "test_{name}".

        Returns:
            The path to the exported repro directory.
        """
    if repro_dir is None:
        repro_dir = os.getcwd()
    repro_dir = os.path.join(repro_dir, 'onnx_debug')
    onnx_graph, onnx_params_dict = _onnx_graph_from_aten_graph(self.graph, self.export_options, self.params_dict)
    proto, _ = _onnx_proto_from_onnx_graph(onnx_graph, self.export_options, onnx_params_dict)
    return OnnxTestCaseRepro.create_test_case_repro(proto, self.input_args, self.pt_outs, repro_dir, name)