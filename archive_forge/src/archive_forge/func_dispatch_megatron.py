import importlib
import warnings
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.init as init
from peft.tuners.tuners_utils import BaseTunerLayer
from .layer import LoraLayer
def dispatch_megatron(target: torch.nn.Module, adapter_name: str, lora_config, **kwargs: Any) -> Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if lora_config.megatron_config:
        megatron_core = importlib.import_module(lora_config.megatron_core)
    else:
        megatron_core = None
    if megatron_core and isinstance(target_base_layer, (megatron_core.tensor_parallel.ColumnParallelLinear, megatron_core.tensor_parallel.RowParallelLinear)):
        megatron_kwargs = kwargs.copy()
        megatron_config = lora_config.megatron_config
        if isinstance(megatron_config, dict):
            transformer_config_class = megatron_core.transformer.transformer_config.TransformerConfig
            megatron_config = transformer_config_class(**lora_config.megatron_config)
        megatron_kwargs['megatron_config'] = megatron_config
        if megatron_kwargs['fan_in_fan_out']:
            warnings.warn('fan_in_fan_out is set to True but the target module is `ColumnParallelLinear` or `RowParallelLinear`. Setting fan_in_fan_out to False.')
            megatron_kwargs['fan_in_fan_out'] = lora_config.fan_in_fan_out = False
        new_module = LoraParallelLinear(base_layer=target, adapter_name=adapter_name, backend=megatron_core.tensor_parallel, **megatron_kwargs)
    return new_module