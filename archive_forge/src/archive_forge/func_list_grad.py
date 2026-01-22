from collections import OrderedDict, defaultdict
import warnings
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, context
from ..context import Context, cpu
from .. import autograd
from .utils import _indent, _brief_print_list, shape_is_known
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def list_grad(self):
    """Returns gradient buffers on all contexts, in the same order
        as :py:meth:`values`."""
    if self._data is not None and self._grad is None:
        raise RuntimeError("Cannot get gradient array for Parameter '%s' because grad_req='null'" % self.name)
    return self._check_and_get(self._grad, list)