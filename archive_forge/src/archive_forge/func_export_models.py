import copy
import gc
import multiprocessing as mp
import os
import traceback
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import onnx
from transformers.modeling_utils import get_parameter_dtype
from transformers.utils import is_tf_available, is_torch_available
from ...onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from ...utils import (
from ...utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ...utils.save_utils import maybe_save_preprocessors
from ..error_utils import AtolError, MinimumVersionError, OutputMatchError, ShapeError
from ..tasks import TasksManager
from .base import OnnxConfig
from .constants import UNPICKABLE_ARCHS
from .model_configs import SpeechT5OnnxConfig
from .utils import (
def export_models(models_and_onnx_configs: Dict[str, Tuple[Union['PreTrainedModel', 'TFPreTrainedModel', 'ModelMixin'], 'OnnxConfig']], output_dir: Path, opset: Optional[int]=None, output_names: Optional[List[str]]=None, device: str='cpu', input_shapes: Optional[Dict]=None, disable_dynamic_axes_fix: Optional[bool]=False, dtype: Optional[str]=None, no_dynamic_axes: bool=False, do_constant_folding: bool=True, model_kwargs: Optional[Dict[str, Any]]=None) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Exports a Pytorch or TensorFlow encoder decoder model to an ONNX Intermediate Representation.
    The following method exports the encoder and decoder components of the model as separate
    ONNX files.

    Args:
        models_and_onnx_configs (`Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`, `ModelMixin`], `OnnxConfig`]]):
            A dictionnary containing the models to export and their corresponding onnx configs.
        output_dir (`Path`):
            Output directory to store the exported ONNX models.
        opset (`Optional[int]`, defaults to `None`):
            The version of the ONNX operator set to use.
        output_names (`Optional[List[str]]`, defaults to `None`):
            The names to use for the exported ONNX files. The order must be the same as the order of submodels in the ordered dict `models_and_onnx_configs`.
            If None, will use the keys from `models_and_onnx_configs` as names.
        device (`str`, defaults to `"cpu"`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes for the example input provided to the ONNX exporter.
        disable_dynamic_axes_fix (`Optional[bool]`, defaults to `False`):
            Whether to disable the default dynamic axes fixing.
        dtype (`Optional[str]`, defaults to `None`):
            Data type to remap the model inputs to. PyTorch-only. Only `fp16` is supported.
        no_dynamic_axes (bool, defaults to `False`):
            If True, disables the use of dynamic axes during ONNX export.
        do_constant_folding (bool, defaults to `True`):
            PyTorch-specific argument. If `True`, the PyTorch ONNX export will fold constants into adjacent nodes, if possible.
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export. This argument should be used along the `custom_onnx_config` argument
            in case, for example, the model inputs/outputs are changed (for example, if
            `model_kwargs={"output_attentions": True}` is passed).
    Returns:
        `Tuple[List[List[str]], List[List[str]]]`: A tuple with an ordered list of the model's inputs, and the named
        outputs from the ONNX configuration.
    """
    outputs = []
    if output_names is not None and len(output_names) != len(models_and_onnx_configs):
        raise ValueError(f'Provided custom names {output_names} for the export of {len(models_and_onnx_configs)} models. Please provide the same number of names as models to export.')
    for i, model_name in enumerate(models_and_onnx_configs.keys()):
        submodel, sub_onnx_config = models_and_onnx_configs[model_name]
        output_name = output_names[i] if output_names is not None else Path(model_name + '.onnx')
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        outputs.append(export(model=submodel, config=sub_onnx_config, output=output_path, opset=opset, device=device, input_shapes=input_shapes, disable_dynamic_axes_fix=disable_dynamic_axes_fix, dtype=dtype, no_dynamic_axes=no_dynamic_axes, do_constant_folding=do_constant_folding, model_kwargs=model_kwargs))
    outputs = list(map(list, zip(*outputs)))
    return outputs