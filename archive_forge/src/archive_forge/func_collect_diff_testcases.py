import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.case.utils import import_recursive
from onnx.onnx_pb import (
def collect_diff_testcases() -> List[TestCase]:
    """Collect node test cases which are different from the main branch"""
    global _DiffOpTypes
    _DiffOpTypes = get_diff_op_types()
    import_recursive(sys.modules[__name__])
    return _NodeTestCases