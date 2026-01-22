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
def validate_models_outputs(models_and_onnx_configs: Dict[str, Tuple[Union['PreTrainedModel', 'TFPreTrainedModel', 'ModelMixin'], 'OnnxConfig']], onnx_named_outputs: List[List[str]], output_dir: Path, atol: Optional[float]=None, onnx_files_subpaths: Optional[List[str]]=None, input_shapes: Optional[Dict]=None, device: str='cpu', use_subprocess: Optional[bool]=True, model_kwargs: Optional[Dict[str, Any]]=None):
    """
    Validates the export of several models, by checking that the outputs from both the reference and the exported model match.
    The following method validates the ONNX models exported using the `export_models` method.

    Args:
        models_and_onnx_configs (`Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]]):
            A dictionnary containing the models to validate and their corresponding onnx configs.
        onnx_named_outputs (`List[List[str]]`):
            The names of the outputs to check.
        output_dir (`Path`):
            Output directory where the exported ONNX models are stored.
        atol (`Optional[float]`, defaults to `None`):
            The absolute tolerance in terms of outputs difference between the reference and the exported model.
        onnx_files_subpaths (`Optional[List[str]]`, defaults to `None`):
            The relative paths from `output_dir` to the ONNX files to do validation on. The order must be the same as the order of submodels
            in the ordered dict `models_and_onnx_configs`. If None, will use the keys from the `models_and_onnx_configs` as names.
        input_shapes (`Optional[Dict]`, defaults to `None`):
            If specified, allows to use specific shapes to validate the ONNX model on.
        device (`str`, defaults to `"cpu"`):
            The device on which the ONNX models will be validated. Either `cpu` or `cuda`. Validation on a CUDA device is supported only for PyTorch.
        use_subprocess (`Optional[bool]`, defaults to `True`):
            Launch validation of each exported model in a subprocess.
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export and validation.
    Raises:
        ValueError: If the outputs shapes or values do not match between the reference and the exported model.
    """
    if len(onnx_named_outputs) != len(models_and_onnx_configs.keys()):
        raise ValueError(f'Invalid number of ONNX named outputs. Required {len(models_and_onnx_configs.keys())}, Provided {len(onnx_named_outputs)}')
    if onnx_files_subpaths is not None and len(onnx_files_subpaths) != len(models_and_onnx_configs):
        raise ValueError(f'Provided custom names {onnx_files_subpaths} for the validation of {len(models_and_onnx_configs)} models. Please provide the same number of ONNX file names as models to export.')
    if use_subprocess:
        logger.info('Validating models in subprocesses...')
    exceptions = []
    for i, model_name in enumerate(models_and_onnx_configs.keys()):
        submodel, sub_onnx_config = models_and_onnx_configs[model_name]
        onnx_model_path = output_dir.joinpath(onnx_files_subpaths[i]) if onnx_files_subpaths is not None else output_dir.joinpath(model_name + '.onnx')
        try:
            validate_model_outputs(config=sub_onnx_config, reference_model=submodel, onnx_model=onnx_model_path, onnx_named_outputs=onnx_named_outputs[i], atol=atol, input_shapes=input_shapes, device=device, use_subprocess=use_subprocess, model_kwargs=model_kwargs)
        except Exception as e:
            exceptions.append((onnx_model_path, e))
    if len(exceptions) != 0:
        for i, exception in enumerate(exceptions[:-1]):
            logger.error(f'Validation for the model {exception[0].as_posix()} raised: {exception[1]}')
        raise exceptions[-1][1]