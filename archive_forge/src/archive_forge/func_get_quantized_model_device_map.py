import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from accelerate.utils.imports import (
from ..big_modeling import dispatch_model, init_empty_weights
from .dataclasses import BnbQuantizationConfig
from .modeling import (
def get_quantized_model_device_map(model, bnb_quantization_config, device_map=None, max_memory=None, no_split_module_classes=None):
    if device_map is None:
        if torch.cuda.is_available():
            device_map = {'': torch.cuda.current_device()}
        else:
            raise RuntimeError('No GPU found. A GPU is needed for quantization.')
        logger.info("The device_map was not initialized.Setting device_map to `{'':torch.cuda.current_device()}`.")
    if isinstance(device_map, str):
        if device_map not in ['auto', 'balanced', 'balanced_low_0', 'sequential']:
            raise ValueError("If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or 'sequential'.")
        special_dtypes = {}
        special_dtypes.update({name: bnb_quantization_config.torch_dtype for name, _ in model.named_parameters() if any((m in name for m in bnb_quantization_config.skip_modules))})
        special_dtypes.update({name: torch.float32 for name, _ in model.named_parameters() if any((m in name for m in bnb_quantization_config.keep_in_fp32_modules))})
        kwargs = {}
        kwargs['special_dtypes'] = special_dtypes
        kwargs['no_split_module_classes'] = no_split_module_classes
        kwargs['dtype'] = bnb_quantization_config.target_dtype
        if device_map != 'sequential':
            max_memory = get_balanced_memory(model, low_zero=device_map == 'balanced_low_0', max_memory=max_memory, **kwargs)
        kwargs['max_memory'] = max_memory
        device_map = infer_auto_device_map(model, **kwargs)
    if isinstance(device_map, dict):
        modules_not_to_convert = bnb_quantization_config.skip_modules + bnb_quantization_config.keep_in_fp32_modules
        device_map_without_some_modules = {key: device_map[key] for key in device_map.keys() if key not in modules_not_to_convert}
        for device in ['cpu', 'disk']:
            if device in device_map_without_some_modules.values():
                if bnb_quantization_config.load_in_4bit:
                    raise ValueError('\n                        Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit\n                        the quantized model. If you want to dispatch the model on the CPU or the disk while keeping\n                        these modules in `torch_dtype`, you need to pass a custom `device_map` to\n                        `load_and_quantize_model`. Check\n                        https://huggingface.co/docs/accelerate/main/en/usage_guides/quantization#offload-modules-to-cpu-and-disk\n                        for more details.\n                        ')
                else:
                    logger.info('Some modules are are offloaded to the CPU or the disk. Note that these modules will be converted to 8-bit')
        del device_map_without_some_modules
    return device_map