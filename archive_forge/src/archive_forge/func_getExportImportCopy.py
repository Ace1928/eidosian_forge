import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_utils import TestCase, \
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from itertools import chain
from typing import List, Union
from torch._C import TensorType
import io
def getExportImportCopy(self, m, also_test_file=True, map_location=None):
    buffer = io.BytesIO()
    torch.jit.save(m, buffer)
    buffer.seek(0)
    imported = torch.jit.load(buffer, map_location=map_location)
    if not also_test_file:
        return imported
    with TemporaryFileName() as fname:
        torch.jit.save(imported, fname)
        return torch.jit.load(fname, map_location=map_location)