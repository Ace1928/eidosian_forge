from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
def traverse_types(self):
    """
        Recursively list all frontend types involved in this model.
        """
    types = [self._fe_type]
    queue = deque([self])
    while len(queue) > 0:
        dm = queue.popleft()
        for i_dm in dm.inner_models():
            if i_dm._fe_type not in types:
                queue.append(i_dm)
                types.append(i_dm._fe_type)
    return types