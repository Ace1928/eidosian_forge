import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
import torch
from torch.jit.generate_bytecode import generate_upgraders_bytecode
from torchgen.code_template import CodeTemplate
from torchgen.operator_versions.gen_mobile_upgraders_constant import (
def sort_upgrader(upgrader_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_upgrader_list = sorted(upgrader_list, key=lambda one_upgrader: next(iter(one_upgrader)))
    return sorted_upgrader_list