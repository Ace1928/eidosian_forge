from collections import deque
from numba.core import types, cgutils
def load_into(self, builder, ptr, formal_list):
    """
        Load the packed values into a sequence indexed by formal
        argument number (skipping any Omitted position).
        """
    self._do_load(builder, ptr, formal_list)