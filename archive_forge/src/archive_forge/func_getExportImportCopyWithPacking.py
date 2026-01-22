from torch.autograd import Variable
from torch.autograd.function import _nested_map
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
from torch.onnx import OperatorExportTypes
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
import zipfile
import functools
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_WINDOWS, \
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from contextlib import contextmanager
from functools import reduce
from io import StringIO
from collections import defaultdict
import importlib.util
import inspect
import io
import math
import os
import pickle
import sys
import tempfile
import textwrap
from importlib.abc import Loader
from typing import Any, Dict, List, Tuple, Union
def getExportImportCopyWithPacking(self, m, also_test_file=True, map_location=None):
    buffer = io.BytesIO()
    m.apply(lambda s: s._pack() if s._c._has_method('_pack') else None)
    torch.jit.save(m, buffer)
    m.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)
    buffer.seek(0)
    imported = torch.jit.load(buffer, map_location=map_location)
    imported.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)
    if not also_test_file:
        return imported
    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        f.close()
        imported.save(f.name)
        result = torch.jit.load(f.name, map_location=map_location)
    finally:
        os.unlink(f.name)
    result.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)
    return result