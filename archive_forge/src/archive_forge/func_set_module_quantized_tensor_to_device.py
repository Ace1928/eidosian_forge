import importlib.metadata
import warnings
from copy import deepcopy
from packaging import version
from ..utils import is_accelerate_available, is_bitsandbytes_available, logging
def set_module_quantized_tensor_to_device(module, tensor_name, device, value=None, quantized_stats=None):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function). The
    function is adapted from `set_module_tensor_to_device` function from accelerate that is adapted to support the
    class `Int8Params` from `bitsandbytes`.

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        quantized_stats (`dict[str, Any]`, *optional*):
            Dict with items for either 4-bit or 8-bit serialization
    """
    if '.' in tensor_name:
        splits = tensor_name.split('.')
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f'{module} has no attribute {split}.')
            module = new_module
        tensor_name = splits[-1]
    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f'{module} does not have a parameter or a buffer named {tensor_name}.')
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)
    if old_value.device == torch.device('meta') and device not in ['meta', torch.device('meta')] and (value is None):
        raise ValueError(f'{tensor_name} is on the meta device, we need a `value` to put in on {device}.')
    prequantized_loading = quantized_stats is not None
    if is_buffer or not is_bitsandbytes_available():
        is_8bit = False
        is_4bit = False
    else:
        is_4bit = hasattr(bnb.nn, 'Params4bit') and isinstance(module._parameters[tensor_name], bnb.nn.Params4bit)
        is_8bit = isinstance(module._parameters[tensor_name], bnb.nn.Int8Params)
    if is_8bit or is_4bit:
        param = module._parameters[tensor_name]
        if param.device.type != 'cuda':
            if value is None:
                new_value = old_value.to(device)
            elif isinstance(value, torch.Tensor):
                new_value = value.to('cpu')
            else:
                new_value = torch.tensor(value, device='cpu')
            if issubclass(module.source_cls, Conv1D) and (not prequantized_loading):
                new_value = new_value.T
            kwargs = old_value.__dict__
            if prequantized_loading != (new_value.dtype in (torch.int8, torch.uint8)):
                raise ValueError(f'Value dtype `{new_value.dtype}` is not compatible with parameter quantization status.')
            if is_8bit:
                is_8bit_serializable = version.parse(importlib.metadata.version('bitsandbytes')) > version.parse('0.37.2')
                if new_value.dtype in (torch.int8, torch.uint8) and (not is_8bit_serializable):
                    raise ValueError('Detected int8 weights but the version of bitsandbytes is not compatible with int8 serialization. Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`.')
                new_value = bnb.nn.Int8Params(new_value, requires_grad=False, **kwargs).to(device)
                if prequantized_loading:
                    setattr(new_value, 'SCB', quantized_stats['SCB'].to(device))
            elif is_4bit:
                if prequantized_loading:
                    is_4bit_serializable = version.parse(importlib.metadata.version('bitsandbytes')) >= version.parse('0.41.3')
                    if new_value.dtype in (torch.int8, torch.uint8) and (not is_4bit_serializable):
                        raise ValueError('Detected 4-bit weights but the version of bitsandbytes is not compatible with 4-bit serialization. Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`.')
                    new_value = bnb.nn.Params4bit.from_prequantized(data=new_value, quantized_stats=quantized_stats, requires_grad=False, device=device, **kwargs)
                else:
                    new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(device)
            module._parameters[tensor_name] = new_value
    else:
        if value is None:
            new_value = old_value.to(device)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)
        if is_buffer:
            module._buffers[tensor_name] = new_value
        else:
            new_value = nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value