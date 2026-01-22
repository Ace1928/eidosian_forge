import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def should_use_int_tensor(op_name: str) -> bool:
    return op_name in should_use_int_tensor_ops_