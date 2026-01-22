import json
import os
from enum import Enum
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D
from transformers.utils.quantization_config import QuantizationMethod
from ..utils import is_accelerate_available, is_auto_gptq_available
from ..utils.modeling_utils import recurse_getattr
from .constants import GPTQ_CONFIG
from .data import get_dataset, prepare_dataset
from .utils import get_block_name_with_pattern, get_device, get_layers, get_preceding_modules, get_seqlen
def load_quantized_model(model: nn.Module, save_folder: str, quant_config_name: str=GPTQ_CONFIG, state_dict_name: Optional[str]=None, device_map: Optional[str]=None, max_memory: Optional[Dict]=None, no_split_module_classes: Optional[Dict]=None, offload_folder: Optional[str]=None, offload_buffers: Optional[str]=None, offload_state_dict: bool=False, disable_exllama: bool=False, exllama_config: Optional[Dict[str, Any]]=None, max_input_length: Optional[int]=None):
    """
    Load quantized weights from the save_folder into the converted model and dispatch the weights according to the device_map.

    Args:
        model (`nn.Module`):
            The model can be enpty or not.
        save_folder (`str`):
            Directory to which to load the weights.
        quant_config_name (`str`, defaults to `GPTQ_CONFIG`):
            Name of the quantization config file
        state_dict_name (`Optional[str]`, defaults to `None`):
            Name of the state dict file
        device_map (`Optional[str]`, defaults to `None`):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`.
        max_memory (`Optional[Dict]`, defaults to `None`):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
        no_split_module_classes (`Optional[Dict]`, defaults to `None`):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        offload_folder (`Optional[str]`, defaults to `None`):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`Optional[str]`, defaults to `None`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        offload_state_dict (`bool`, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.
        disable_exllama (`Optional[bool]`, defaults to `None`):
            Whether to use exllama backend. Only works with `bits` = 4.
        exllama_config (`Optional[Dict[str, Any]]`, defaults to `None`):
            The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults to `{"version": 2}` if unset.
        max_input_length (`Optional[int]`, defaults to `None`):
            The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input length.
            It is specific to the exllama backend with act-order.

    Returns:
        `nn.Module`: The quantized model
    """
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU found. A GPU is needed to run quantized model.')
    if not is_auto_gptq_available():
        raise RuntimeError('auto-gptq is required in order to load quantized weights : `pip install auto-gptq`')
    if not is_accelerate_available():
        raise RuntimeError('You need to install accelerate in order to load and dispatch weights toa quantized model. You can do it with `pip install accelerate`')
    if device_map is None:
        device_map = {'': torch.cuda.current_device()}
        logger.info("The device_map was not initialized.Setting device_map to `{'':torch.cuda.current_device()}`.")
    if exllama_config is None:
        exllama_config = {'version': ExllamaVersion.TWO}
    elif 'version' not in exllama_config:
        raise ValueError('`exllama_config` needs to have a `version` key')
    elif exllama_config['version'] not in [ExllamaVersion.ONE, ExllamaVersion.TWO]:
        version = exllama_config['version']
        raise ValueError(f'Only supported versions are in [ExllamaVersion.ONE, ExllamaVersion.TWO] - not recognized version {version}')
    try:
        if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
            quantize_config_dict = model.config.quantization_config.to_dict()
        else:
            with open(os.path.join(save_folder, quant_config_name), 'r', encoding='utf-8') as f:
                quantize_config_dict = json.load(f)
    except Exception as err:
        raise ValueError(f"Failed to load quantization config from {save_folder} (lookup for traceback): {err}\nTip: If the save directory is saved from a transformers.PreTrainedModel, make sure that `config.json` contains a 'quantization_config' key.") from err
    quantizer = GPTQQuantizer.from_dict(quantize_config_dict)
    quantizer.disable_exllama = disable_exllama
    quantizer.exllama_config = exllama_config
    quantizer.exllama_version = quantizer.exllama_config['version']
    quantizer.max_input_length = max_input_length
    model = quantizer.convert_model(model)
    if no_split_module_classes is None:
        no_split_module_classes = quantizer.get_no_split_module_classes(model)
    model = load_checkpoint_and_dispatch(model, checkpoint=os.path.join(save_folder, state_dict_name) if state_dict_name is not None else save_folder, device_map=device_map, max_memory=max_memory, no_split_module_classes=no_split_module_classes, offload_folder=offload_folder, offload_buffers=offload_buffers, offload_state_dict=offload_state_dict)
    model = quantizer.post_init_model(model)
    model.is_quantized = True
    model.quantization_method = QuantizationMethod.GPTQ
    model.eval()
    return model