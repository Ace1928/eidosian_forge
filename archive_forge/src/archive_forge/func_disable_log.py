from torch import _C
from torch._C import _onnx as _C_onnx
from torch._C._onnx import (
from . import (  # usort:skip. Keep the order instead of sorting lexicographically
from ._exporter_states import ExportTypes, SymbolicContext
from ._type_utils import JitScalarType
from .errors import CheckerError  # Backwards compatibility
from .utils import (
from ._internal.exporter import (  # usort:skip. needs to be last to avoid circular import
from ._internal.onnxruntime import (
def disable_log() -> None:
    """Disables ONNX logging."""
    _C._jit_set_onnx_log_enabled(False)