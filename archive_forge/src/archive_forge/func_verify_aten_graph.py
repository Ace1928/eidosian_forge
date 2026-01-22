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
def verify_aten_graph(graph: torch.Graph, input_args: Tuple[Any, ...], export_options: _experimental.ExportOptions, params_dict: Optional[Dict[str, Any]]=None, verification_options: Optional[VerificationOptions]=None) -> Tuple[Optional[AssertionError], torch.Graph, _OutputsType, _OutputsType]:
    if verification_options is None:
        verification_options = VerificationOptions()
    if params_dict is None:
        params_dict = {}
    original_jit_graph = graph
    graph = graph.copy()
    graph_inputs = list(graph.inputs())
    jit_inputs = tuple([arg for arg in input_args if arg is not None])
    weights = [params_dict[v.debugName()] for v in graph_inputs[len(jit_inputs):]]
    assert all((w is not None for w in weights))
    jit_inputs = copy.deepcopy(jit_inputs)
    jit_input_and_parameters = jit_inputs + tuple(weights)
    jit_outs = torch._C._jit_interpret_graph(graph, jit_input_and_parameters)
    if not isinstance(jit_outs, (list, tuple)):
        jit_outs = [jit_outs]
    graph, onnx_params_dict = _onnx_graph_from_aten_graph(graph, export_options, params_dict)
    proto, export_map = _onnx_proto_from_onnx_graph(graph, export_options, onnx_params_dict)
    model_f: Union[str, io.BytesIO] = io.BytesIO()
    export_type = _exporter_states.ExportTypes.PROTOBUF_FILE
    onnx_proto_utils._export_file(proto, model_f, export_type, export_map)
    try:
        new_input_names = {v.debugName() for v in graph.inputs()}
        new_input_args = []
        for v, arg in zip(original_jit_graph.inputs(), input_args):
            if v.debugName() in new_input_names:
                new_input_args.append(arg)
        input_args = tuple(new_input_args)
        onnx_inputs = _prepare_input_for_onnx(input_args, {}, verification_options.remained_onnx_input_idx, verification_options.flatten)
        onnx_session = _onnx_backend_session(model_f, verification_options.backend)
        onnx_outs = _run_onnx(onnx_session, onnx_inputs)
        del onnx_session
        try:
            _compare_onnx_pytorch_outputs(onnx_outs=onnx_outs, pt_outs=jit_outs, options=verification_options)
        except AssertionError as e:
            return (e, graph, jit_outs, onnx_outs)
        return (None, graph, jit_outs, onnx_outs)
    except Exception as e:
        print('Unexpected error during verification.')
        print('jit graph: ', original_jit_graph)
        print('onnx graph: ', graph)
        raise e