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
@classmethod
def assert_similar_outputs(cls, ref_outputs: Sequence[Any], outputs: Sequence[Any], rtol: float, atol: float, model_dir: str | None=None) -> None:
    try:
        np.testing.assert_equal(len(outputs), len(ref_outputs))
    except TypeError as e:
        raise TypeError(f'Unable to compare expected type {type(ref_outputs)} and runtime type {type(outputs)} (known test={model_dir or '?'!r})') from e
    for i in range(len(outputs)):
        if isinstance(outputs[i], (list, tuple)):
            if not isinstance(ref_outputs[i], (list, tuple)):
                raise AssertionError(f'Unexpected type {type(outputs[i])} for outputs[{i}]. Expected type is {type(ref_outputs[i])} (known test={model_dir or '?'!r}).')
            for j in range(len(outputs[i])):
                cls.assert_similar_outputs(ref_outputs[i][j], outputs[i][j], rtol, atol, model_dir=model_dir)
        else:
            if not np.issubdtype(ref_outputs[i].dtype, np.number):
                if ref_outputs[i].tolist() != outputs[i].tolist():
                    raise AssertionError(f'{ref_outputs[i]} != {outputs[i]}')
                continue
            np.testing.assert_equal(outputs[i].dtype, ref_outputs[i].dtype)
            if ref_outputs[i].dtype == object:
                np.testing.assert_array_equal(outputs[i], ref_outputs[i])
            else:
                np.testing.assert_allclose(outputs[i], ref_outputs[i], rtol=rtol, atol=atol)