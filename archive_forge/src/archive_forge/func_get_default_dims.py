from collections.abc import Sequence
import numpy as np
import xarray as xr
from numpy.linalg import LinAlgError
from scipy import special, stats
from . import _remove_indexes_to_reduce
from .linalg import cholesky, eigh
def get_default_dims(dims):
    """Get default dims on which to perfom an operation.

    Whenever a function from :mod:`xarray_einstats.stats` is called with
    ``dims=None`` (the default) this function is called to choose the
    default dims on which to operate out of the list with all the dims present.

    This function is thought to be monkeypatched by domain specific applications
    as shown in the examples.

    Parameters
    ----------
    dims : list of str
        List with all the dimensions of the input DataArray in the order they
        appear.

    Returns
    -------
    list of str
        List with the dimensions on which to apply the operation.
        ``xarray_einstats`` defaults to applying the operation to all
        dimensions. Monkeypatch this function to get a different result.

    Examples
    --------
    The ``xarray_einstats`` default behaviour is operating (averaging in this case)
    over all dimensions present in the input DataArray:

    .. jupyter-execute::

        from xarray_einstats import stats, tutorial
        da = tutorial.generate_mcmc_like_dataset(3)["mu"]
        stats.hmean(da)

    Here we show how to monkeypatch ``get_default_dims`` to get a different default
    behaviour. If you use ``xarray_einstats`` and {doc}`arviz:index` to work
    with MCMC results, operating over chain and dim only might be a better default:

    .. jupyter-execute::

        def get_default_dims(dims):
            out = [dim for dim in ("chain", "draw") if dim in dims]
            if not out:  # if chain nor draw are present fall back to all dims
                return dims
            return out
        stats.get_default_dims = get_default_dims
        stats.hmean(da)

    You can still use ``dims`` explicitly to average over any custom dimension

    .. jupyter-execute::

        stats.hmean(da, dims="team")

    """
    return dims