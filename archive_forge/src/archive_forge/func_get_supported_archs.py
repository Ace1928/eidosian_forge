import importlib
from typing import List, Optional, Type
import torch.nn as nn
from vllm.logger import init_logger
from vllm.utils import is_hip, is_neuron
@staticmethod
def get_supported_archs() -> List[str]:
    return list(_MODELS.keys())