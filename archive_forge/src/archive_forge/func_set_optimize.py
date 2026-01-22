import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def set_optimize(self, for_size=None, combine_backing_indices=None):
    """Change how the builder tries to optimize the result.

        :param for_size: Tell the builder to try and make the index as small as
            possible.
        :param combine_backing_indices: If the builder spills to disk to save
            memory, should the on-disk indices be combined. Set to True if you
            are going to be probing the index, but to False if you are not. (If
            you are not querying, then the time spent combining is wasted.)
        :return: None
        """
    if for_size is not None:
        self._optimize_for_size = for_size
    if combine_backing_indices is not None:
        self._combine_backing_indices = combine_backing_indices