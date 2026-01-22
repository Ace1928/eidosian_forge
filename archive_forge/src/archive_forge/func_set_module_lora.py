from typing import List, Optional
import torch
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
def set_module_lora(self, module_name: str, lora: LoRALayerWeights):
    self._loras[module_name] = lora