from __future__ import annotations
import os
import sys
from typing import Any, Iterable
import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.onnx_cpp2py_export.checker as c_checker
def check_large_initializers(self):
    for tensor in ext_data._get_all_tensors(self.model_proto):
        if not ext_data.uses_external_data(tensor):
            continue
        prop: onnx.StringStringEntryProto | None = None
        for ext in tensor.external_data:
            if ext.key == 'location':
                prop = ext
        if prop is None:
            raise RuntimeError(f'No location found for tensor name {tensor.name!r}.')
        if prop.value not in self.large_initializers:
            raise RuntimeError(f'Unable to find large tensor named {tensor.name!r} with location {prop.value!r} in {sorted(self.large_initializers)}.')