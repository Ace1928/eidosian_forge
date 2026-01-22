imported when doing ``import xarray_einstats``.
import warnings
from collections.abc import Hashable
import einops
import xarray as xr
def raw_rearrange(*args, **kwargs):
    """Wrap einops.rearrange.

    DEPRECATED
    """
    warnings.warn('raw_rearrange has been deprecated. Its functionality has been merged into rearrange', DeprecationWarning)
    return rearrange(*args, **kwargs)