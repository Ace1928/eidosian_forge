from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
def begin_state(self, **kwargs):
    assert not self._modified, 'After applying modifier cells (e.g. DropoutCell) the base cell cannot be called directly. Call the modifier cell instead.'
    return _cells_begin_state(self._children.values(), **kwargs)