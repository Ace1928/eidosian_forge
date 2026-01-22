import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def should_check_resize(schema: FunctionSchema) -> bool:
    schema_str = str(schema)
    type_variant_op_name = schema_str[:schema_str.find('(')]
    return type_variant_op_name not in no_memory_resize_ops