import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from packaging import version
from ..utils import is_auto_awq_available, is_torch_available, logging
def quantization_method(self):
    """
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
    if self.load_in_8bit:
        return 'llm_int8'
    elif self.load_in_4bit and self.bnb_4bit_quant_type == 'fp4':
        return 'fp4'
    elif self.load_in_4bit and self.bnb_4bit_quant_type == 'nf4':
        return 'nf4'
    else:
        return None