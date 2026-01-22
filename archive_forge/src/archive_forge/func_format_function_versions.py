import os
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Sequence, Set, Tuple
import numpy as np
from onnx import defs, helper
from onnx.backend.sample.ops import collect_sample_implementations
from onnx.backend.test.case import collect_snippets
from onnx.defs import ONNX_ML_DOMAIN, OpSchema
def format_function_versions(function_versions: Sequence[int]) -> str:
    return f'{', '.join([str(v) for v in function_versions])}'