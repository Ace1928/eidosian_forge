import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def get_out_kernel_name(g: NativeFunctionsGroup, backend_index: BackendIndex) -> str:
    kernel = backend_index.get_kernel(g.out)
    if g.structured or kernel is None:
        return cpp.name(g.out.func)
    return kernel.kernel