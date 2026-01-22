import copy
import enum
import gc
import inspect
import itertools
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import onnx
from transformers.utils import is_accelerate_available, is_torch_available
from ...onnx import remove_duplicate_weights_from_tied_info
from ...onnx import merge_decoders
from ...utils import (
from ...utils import TORCH_MINIMUM_VERSION as GLOBAL_MIN_TORCH_VERSION
from ...utils import TRANSFORMERS_MINIMUM_VERSION as GLOBAL_MIN_TRANSFORMERS_VERSION
from ...utils.doc import add_dynamic_docstring
from ...utils.import_utils import check_if_transformers_greater, is_onnx_available, is_onnxruntime_available
from ..base import ExportConfig
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import ModelPatcher, Seq2SeqModelPatcher
def fix_dynamic_axes(self, model_path: 'Path', device: str='cpu', dtype: Optional[str]=None, input_shapes: Optional[Dict]=None):
    """
        Fixes potential issues with dynamic axes.
        During the export, ONNX will infer some axes to be dynamic which are actually static. This method is called
        right after the export to fix such issues.

        Args:
            model_path (`Path`):
                The path of the freshly exported ONNX model.
        """
    if not (is_onnx_available() and is_onnxruntime_available()):
        raise RuntimeError('The onnx and onnxruntime packages are necessary to fix the dynamic shapes of the exported model. You can install them by doing: pip install onnx onnxruntime')
    import onnx
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
    allowed_dynamic_axes = set()
    for input_ in self.inputs.values():
        allowed_dynamic_axes |= set(input_.values())
    for output in self.outputs.values():
        allowed_dynamic_axes |= set(output.values())
    if device.startswith('cuda'):
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    session_options = SessionOptions()
    session_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    session = InferenceSession(model_path.as_posix(), providers=providers, sess_options=session_options)
    onnx_input_names = [inp.name for inp in session.get_inputs()]
    to_fix = []
    for output_idx, node in enumerate(session.get_outputs()):
        for idx, axis in enumerate(node.shape):
            if isinstance(axis, str) and axis not in allowed_dynamic_axes:
                to_fix.append((output_idx, idx))
    if to_fix:
        if input_shapes is None:
            input_shapes = {}
        dummy_inputs = self.generate_dummy_inputs(framework='np', **input_shapes)
        dummy_inputs = self.generate_dummy_inputs_for_validation(dummy_inputs, onnx_input_names=onnx_input_names)
        onnx_inputs = {}
        for name, value in dummy_inputs.items():
            if isinstance(value, (list, tuple)):
                value = self.flatten_output_collection_property(name, value)
                onnx_inputs.update(dict(value.items()))
            else:
                onnx_inputs[name] = value
        for name, value in onnx_inputs.items():
            if value.dtype == np.float32 and dtype == 'fp16':
                onnx_inputs[name] = onnx_inputs[name].astype(np.float16)
        outputs = session.run(None, onnx_inputs)
        del session
        onnx_model = onnx.load(model_path.as_posix(), load_external_data=False)
        for output_idx, dim_idx in to_fix:
            dims = onnx_model.graph.output[output_idx].type.tensor_type.shape.dim
            dims[dim_idx].dim_value = outputs[output_idx].shape[dim_idx]
        onnx.save(onnx_model, model_path.as_posix())
        del onnx_model
        gc.collect()