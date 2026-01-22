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
def setattr(self, name, value):
    """Set an attribute to a new value for all Parameters.

        For example, set grad_req to null if you don't need gradient w.r.t a
        model's Parameters::

            model.collect_params().setattr('grad_req', 'null')

        or change the learning rate multiplier::

            model.collect_params().setattr('lr_mult', 0.5)

        Parameters
        ----------
        name : str
            Name of the attribute.
        value : valid type for attribute name
            The new value for the attribute.
        """
    for i in self.values():
        setattr(i, name, value)