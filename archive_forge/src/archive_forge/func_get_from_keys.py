from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
from vllm.model_executor.layers.linear import LinearMethodBase
@staticmethod
def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:
    """Get a value from the model's quantization config."""
    for key in keys:
        if key in config:
            return config[key]
    raise ValueError(f"Cannot find any of {keys} in the model's quantization config.")