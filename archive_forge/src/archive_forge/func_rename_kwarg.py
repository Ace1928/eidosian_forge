imported when doing ``import xarray_einstats``.
import warnings
from collections.abc import Hashable
import einops
import xarray as xr
def rename_kwarg(self, key):
    """Process kwargs for axes_lengths.

        Users use as keys the dimension names they used in the input expressions
        which need to be converted and use the placeholder as key when passed
        to einops functions.
        """
    return self.mapping.get(key, key)