from typing import Any, Optional
import torch
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import get_auto_gptq_quant_linear
def dispatch_gptq(target: torch.nn.Module, adapter_name: str, **kwargs: Any) -> Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    gptq_quantization_config = kwargs.get('gptq_quantization_config', None)
    AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)
    if AutoGPTQQuantLinear is not None and isinstance(target_base_layer, AutoGPTQQuantLinear):
        new_module = QuantLinear(target, adapter_name, **kwargs)
        target.qweight = target_base_layer.qweight
    return new_module