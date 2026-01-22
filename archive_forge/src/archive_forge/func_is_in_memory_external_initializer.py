from __future__ import annotations
import os
import sys
from typing import Any, Iterable
import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.onnx_cpp2py_export.checker as c_checker
def is_in_memory_external_initializer(self, name: str) -> bool:
    """Tells if an initializer name is an external initializer stored in memory.
        The name must start with '#' in that case.
        """
    return name.startswith('#')