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
def onnx_export_from_model(model: Union['PreTrainedModel', 'TFPreTrainedModel'], output: Union[str, Path], opset: Optional[int]=None, optimize: Optional[str]=None, monolith: bool=False, no_post_process: bool=False, atol: Optional[float]=None, do_validation: bool=True, model_kwargs: Optional[Dict[str, Any]]=None, custom_onnx_configs: Optional[Dict[str, 'OnnxConfig']]=None, fn_get_submodels: Optional[Callable]=None, _variant: str='default', legacy: bool=False, preprocessors: List=None, device: str='cpu', no_dynamic_axes: bool=False, task: Optional[str]=None, use_subprocess: bool=False, do_constant_folding: bool=True, **kwargs_shapes):
    """
    Full-suite ONNX export function, exporting **from a pre-loaded PyTorch or Tensorflow model**. This function is especially useful in case one needs to do modifications on the model, as overriding a forward call, before exporting to ONNX.

    Args:
        > Required parameters

        model (`Union["PreTrainedModel", "TFPreTrainedModel"]`):
            PyTorch or TensorFlow model to export to ONNX.
        output (`Union[str, Path]`):
            Path indicating the directory where to store the generated ONNX model.

        > Optional parameters

        task (`Optional[str]`, defaults to `None`):
            The task to export the model for. If not specified, the task will be auto-inferred based on the model.
        opset (`Optional[int]`, defaults to `None`):
            If specified, ONNX opset version to export the model with. Otherwise, the default opset for the given model architecture
            will be used.
        device (`str`, defaults to `"cpu"`):
            The device to use to do the export. Defaults to "cpu".
        optimize (`Optional[str]`, defaults to `None`):
            Allows to run ONNX Runtime optimizations directly during the export. Some of these optimizations are specific to
            ONNX Runtime, and the resulting ONNX will not be usable with other runtime as OpenVINO or TensorRT.
            Available options: `"O1", "O2", "O3", "O4"`. Reference: [`~optimum.onnxruntime.AutoOptimizationConfig`]
        monolith (`bool`, defaults to `False`):
            Forces to export the model as a single ONNX file.
        no_post_process (`bool`, defaults to `False`):
            Allows to disable any post-processing done by default on the exported ONNX models.
        atol (`Optional[float]`, defaults to `None`):
            If specified, the absolute difference tolerance when validating the model. Otherwise, the default atol for the model will be used.
        model_kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            Experimental usage: keyword arguments to pass to the model during
            the export. This argument should be used along the `custom_onnx_configs` argument
            in case, for example, the model inputs/outputs are changed (for example, if
            `model_kwargs={"output_attentions": True}` is passed).
        custom_onnx_configs (`Optional[Dict[str, OnnxConfig]]`, defaults to `None`):
            Experimental usage: override the default ONNX config used for the given model. This argument may be useful for advanced users that desire a finer-grained control on the export. An example is available [here](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model).
        fn_get_submodels (`Optional[Callable]`, defaults to `None`):
            Experimental usage: Override the default submodels that are used at the export. This is
            especially useful when exporting a custom architecture that needs to split the ONNX (e.g. encoder-decoder). If unspecified with custom models, optimum will try to use the default submodels used for the given task, with no guarantee of success.
        use_subprocess (`bool`, defaults to `False`):
            Do the ONNX exported model validation in subprocesses. This is especially useful when
            exporting on CUDA device, where ORT does not release memory at inference session
            destruction. When set to `True`, the `main_export` call should be guarded in
            `if __name__ == "__main__":` block.
        _variant (`str`, defaults to `default`):
            Specify the variant of the ONNX export to use.
        legacy (`bool`, defaults to `False`):
            Disable the use of position_ids for text-generation models that require it for batched generation. Also enable to export decoder only models in three files (without + with past and the merged model). This argument is introduced for backward compatibility and will be removed in a future release of Optimum.
        no_dynamic_axes (bool, defaults to `False`):
            If True, disables the use of dynamic axes during ONNX export.
        do_constant_folding (bool, defaults to `True`):
            PyTorch-specific argument. If `True`, the PyTorch ONNX export will fold constants into adjacent nodes, if possible.
        **kwargs_shapes (`Dict`):
            Shapes to use during inference. This argument allows to override the default shapes used during the ONNX export.

    Example usage:
    ```python
    >>> from transformers import AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> # At this point, we could override some submodules, forward methods, weights, etc. from the model.

    >>> onnx_export_from_model(model, output="gpt2_onnx/")
    ```
    """
    library_name = TasksManager._infer_library_from_model(model)
    TasksManager.standardize_model_attributes(model, library_name)
    if hasattr(model.config, 'export_model_type'):
        model_type = model.config.export_model_type.replace('_', '-')
    else:
        model_type = model.config.model_type.replace('_', '-')
    custom_architecture = library_name == 'transformers' and model_type not in TasksManager._SUPPORTED_MODEL_TYPE
    if task is not None:
        task = TasksManager.map_from_synonym(task)
    else:
        try:
            task = TasksManager._infer_task_from_model_or_model_class(model=model)
        except (ValueError, KeyError) as e:
            raise RuntimeError(f'The model task could not be automatically inferred in `onnx_export_from_model`. Please provide the argument `task` with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}')
        if not custom_architecture and library_name != 'diffusers' and (task + '-with-past' in TasksManager.get_supported_tasks_for_model_type(model_type, 'onnx', library_name=library_name)) and (not monolith):
            task = task + '-with-past'
        logger.info(f'Automatic task detection to: {task}.')
    framework = 'pt' if is_torch_available() and isinstance(model, torch.nn.Module) else 'tf'
    dtype = get_parameter_dtype(model) if framework == 'pt' else model.dtype
    if 'bfloat16' in str(dtype):
        float_dtype = 'bf16'
    elif 'float16' in str(dtype):
        float_dtype = 'fp16'
    else:
        float_dtype = 'fp32'
    if custom_architecture and custom_onnx_configs is None:
        raise ValueError(f'Trying to export a {model_type} model, that is a custom or unsupported architecture, but no custom onnx configuration was passed as `custom_onnx_configs`. Please refer to https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#custom-export-of-transformers-models for an example on how to export custom models. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the model type {model_type} to be supported natively in the ONNX export.')
    if task.startswith('text-generation') and model.config.is_encoder_decoder:
        raise ValueError(f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug reportat https://github.com/huggingface/optimum, if --task was explicitely passed, make sure you selected the right task for the model, referring to `optimum.exporters.tasks.TaskManager`'s `_TRANSFORMERS_TASKS_TO_MODEL_LOADERS`.")
    if legacy and model_type in MODEL_TYPES_REQUIRING_POSITION_IDS and task.startswith('text-generation'):
        logger.warning(f'legacy=True was specified in the ONNX export, although the model {model_type} requires position_ids for batched inference. Passing `legacy=True` is strongly discouraged, and this option will be removed in a future release. Reference: https://github.com/huggingface/optimum/pull/1381')
    if library_name != 'diffusers' and model_type in TasksManager._UNSUPPORTED_CLI_MODEL_TYPE:
        raise ValueError(f'{model_type} is not supported yet. Only {list(TasksManager._SUPPORTED_CLI_MODEL_TYPE.keys())} are supported. If you want to support {model_type} please propose a PR or open up an issue.')
    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True)
    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        input_shapes[input_name] = kwargs_shapes[input_name] if input_name in kwargs_shapes else DEFAULT_DUMMY_SHAPES[input_name]
        if model_type in MODEL_TO_PATCH_FOR_PAST and input_name == 'sequence_length' and (kwargs_shapes.get(input_name) == 1):
            raise ValueError(f'Exporting with a sequence length of 1 a {model_type} model is not supported and can yield unexpected results.')
    onnx_config, models_and_onnx_configs = _get_submodels_and_onnx_configs(model=model, task=task, monolith=monolith, custom_onnx_configs=custom_onnx_configs if custom_onnx_configs is not None else {}, custom_architecture=custom_architecture, float_dtype=float_dtype, fn_get_submodels=fn_get_submodels, preprocessors=preprocessors, _variant=_variant, legacy=legacy, library_name=library_name, model_kwargs=model_kwargs)
    if library_name != 'diffusers':
        if opset is None:
            opset = onnx_config.DEFAULT_ONNX_OPSET
        elif opset < onnx_config.DEFAULT_ONNX_OPSET:
            logger.warning(f'Opset {opset} is lower than the recommended minmum opset ({onnx_config.DEFAULT_ONNX_OPSET}) to export {model_type}. The ONNX export may fail or the exported model may be suboptimal.')
        if atol is None:
            atol = onnx_config.ATOL_FOR_VALIDATION
            if isinstance(atol, dict):
                atol = atol[task.replace('-with-past', '')]
        model.config.save_pretrained(output)
        generation_config = getattr(model, 'generation_config', None)
        if generation_config is not None:
            generation_config.save_pretrained(output)
        model_name_or_path = model.config._name_or_path
        maybe_save_preprocessors(model_name_or_path, output)
        onnx_files_subpaths = [key + '.onnx' for key in models_and_onnx_configs.keys()]
    else:
        for model_name in models_and_onnx_configs:
            subcomponent = models_and_onnx_configs[model_name][0]
            if hasattr(subcomponent, 'save_config'):
                subcomponent.save_config(output / model_name)
            elif hasattr(subcomponent, 'config') and hasattr(subcomponent.config, 'save_pretrained'):
                subcomponent.config.save_pretrained(output / model_name)
        onnx_files_subpaths = [os.path.join(name_dir, ONNX_WEIGHTS_NAME) for name_dir in models_and_onnx_configs]
        model.scheduler.save_pretrained(output.joinpath('scheduler'))
        feature_extractor = getattr(model, 'feature_extractor', None)
        if feature_extractor is not None:
            feature_extractor.save_pretrained(output.joinpath('feature_extractor'))
        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output.joinpath('tokenizer'))
        tokenizer_2 = getattr(model, 'tokenizer_2', None)
        if tokenizer_2 is not None:
            tokenizer_2.save_pretrained(output.joinpath('tokenizer_2'))
        model.save_config(output)
    if float_dtype == 'bf16':
        logger.warning(f'Exporting the model {model.__class__.__name__} in bfloat16 float dtype. After the export, ONNX Runtime InferenceSession with CPU/CUDA execution provider likely does not implement all operators for the bfloat16 data type, and the loading is likely to fail.')
    _, onnx_outputs = export_models(models_and_onnx_configs=models_and_onnx_configs, opset=opset, output_dir=output, output_names=onnx_files_subpaths, input_shapes=input_shapes, device=device, dtype=float_dtype, no_dynamic_axes=no_dynamic_axes, do_constant_folding=do_constant_folding, model_kwargs=model_kwargs)
    if optimize is not None:
        from ...onnxruntime import AutoOptimizationConfig, ORTOptimizer
        optimizer = ORTOptimizer.from_pretrained(output, file_names=onnx_files_subpaths)
        optimization_config = AutoOptimizationConfig.with_optimization_level(optimization_level=optimize)
        optimization_config.disable_shape_inference = True
        optimizer.optimize(save_dir=output, optimization_config=optimization_config, file_suffix='')
    if not no_post_process and library_name != 'diffusers':
        try:
            logger.info('Post-processing the exported models...')
            models_and_onnx_configs, onnx_files_subpaths = onnx_config.post_process_exported_models(output, models_and_onnx_configs, onnx_files_subpaths)
        except Exception as e:
            raise Exception(f'The post-processing of the ONNX export failed. The export can still be performed by passing the option --no-post-process. Detailed error: {e}')
    if library_name == 'diffusers':
        use_subprocess = False
    elif model_type in UNPICKABLE_ARCHS:
        use_subprocess = False
    if device == 'cpu':
        use_subprocess = False
    if do_validation is True:
        try:
            validate_models_outputs(models_and_onnx_configs=models_and_onnx_configs, onnx_named_outputs=onnx_outputs, atol=atol, output_dir=output, onnx_files_subpaths=onnx_files_subpaths, input_shapes=input_shapes, device=device, use_subprocess=use_subprocess, model_kwargs=model_kwargs)
            logger.info(f'The ONNX export succeeded and the exported model was saved at: {output.as_posix()}')
        except ShapeError as e:
            raise e
        except AtolError as e:
            logger.warning(f'The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}')
        except OutputMatchError as e:
            logger.warning(f'The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}')
        except Exception as e:
            raise Exception(f'An error occured during validation, but the model was saved nonetheless at {output.as_posix()}. Detailed error: {e}.')