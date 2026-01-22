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
@grad_req.setter
def grad_req(self, req):
    if req != 'null':
        warnings.warn('Constant parameter "{}" does not support grad_req other than "null", and new value "{}" is ignored.'.format(self.name, req))