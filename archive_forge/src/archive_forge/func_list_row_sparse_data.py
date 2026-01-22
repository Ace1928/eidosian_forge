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
def list_row_sparse_data(self, row_id):
    """Returns copies of the 'row_sparse' parameter on all contexts, in the same order
        as creation. The copy only retains rows whose ids occur in provided row ids.
        The parameter must have been initialized before.

        Parameters
        ----------
        row_id: NDArray
            Row ids to retain for the 'row_sparse' parameter.

        Returns
        -------
        list of NDArrays
        """
    if self._stype != 'row_sparse':
        raise RuntimeError("Cannot return copies of Parameter '%s' on all contexts via list_row_sparse_data() because its storage type is %s. Please use data() instead." % (self.name, self._stype))
    return self._get_row_sparse(self._data, list, row_id)