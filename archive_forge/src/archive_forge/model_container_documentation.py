from __future__ import annotations
import os
import sys
from typing import Any, Iterable
import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.onnx_cpp2py_export.checker as c_checker
Loads large initializers.

        Arguments:
            file_path: model file, the weight are expected to be in the same folder as this file
        