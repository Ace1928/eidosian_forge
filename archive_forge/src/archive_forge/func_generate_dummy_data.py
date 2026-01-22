from __future__ import annotations
import functools
import glob
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import unittest
from collections import defaultdict
from typing import Any, Callable, Iterable, Pattern, Sequence
from urllib.request import urlretrieve
import numpy as np
import onnx
import onnx.reference
from onnx import ONNX_ML, ModelProto, NodeProto, TypeProto, ValueInfoProto, numpy_helper
from onnx.backend.base import Backend
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner.item import TestItem
@staticmethod
def generate_dummy_data(x: ValueInfoProto, seed: int=0, name: str='', random: bool=False) -> np.ndarray:
    """Generates a random tensor based on the input definition."""
    if not x.type.tensor_type:
        raise NotImplementedError(f'Input expected to have tensor type. Unable to generate random data for model {name!r} and input {x}.')
    if x.type.tensor_type.elem_type != 1:
        raise NotImplementedError(f'Currently limited to float tensors. Unable to generate random data for model {name!r} and input {x}.')
    shape = tuple((d.dim_value if d.HasField('dim_value') else 1 for d in x.type.tensor_type.shape.dim))
    if random:
        gen = np.random.default_rng(seed=seed)
        return gen.random(shape, np.float32)
    n = np.prod(shape)
    return (np.arange(n).reshape(shape) / n).astype(np.float32)