import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
def create_quantized_param(self, model: 'PreTrainedModel', param_value: 'torch.Tensor', param_name: str, target_device: 'torch.device', state_dict: Dict[str, Any], unexpected_keys: List[str]):
    """
        combines logic from _load_state_dict_into_meta_model and .integrations.bitsandbytes.py::set_module_quantized_tensor_to_device()
        """
    import bitsandbytes as bnb
    module, tensor_name = get_module_from_name(model, param_name)
    if tensor_name not in module._parameters:
        raise ValueError(f'{module} does not have a parameter or a buffer named {tensor_name}.')
    old_value = getattr(module, tensor_name)
    if tensor_name == 'bias':
        if param_value is None:
            new_value = old_value.to(target_device)
        else:
            new_value = param_value.to(target_device)
        new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad)
        module._parameters[tensor_name] = new_value
        return
    if not isinstance(module._parameters[tensor_name], bnb.nn.Params4bit):
        raise ValueError('this function only loads `Linear4bit components`')
    if old_value.device == torch.device('meta') and target_device not in ['meta', torch.device('meta')] and (param_value is None):
        raise ValueError(f'{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.')
    if self.pre_quantized:
        if not self.is_serializable:
            raise ValueError('Detected int4 weights but the version of bitsandbytes is not compatible with int4 serialization. Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`.')
        if param_name + '.quant_state.bitsandbytes__fp4' not in state_dict and param_name + '.quant_state.bitsandbytes__nf4' not in state_dict:
            raise ValueError(f'Supplied state dict for {param_name} does not contain `bitsandbytes__*` and possibly other `quantized_stats` components.')
        quantized_stats = {}
        for k, v in state_dict.items():
            if param_name + '.' in k:
                quantized_stats[k] = v
                unexpected_keys.remove(k)
        new_value = bnb.nn.Params4bit.from_prequantized(data=param_value, quantized_stats=quantized_stats, requires_grad=False, device=target_device)
    else:
        new_value = param_value.to('cpu')
        if issubclass(module.source_cls, Conv1D):
            new_value = new_value.T
        kwargs = old_value.__dict__
        new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(target_device)
    module._parameters[tensor_name] = new_value