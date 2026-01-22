import numpy as np
from pygsp import filters, utils
@utils.filterbank_handler
def compute_norm_tig(g, **kwargs):
    """
    Compute the :math:`\\ell_2` norm of the Tig.
    See :func:`compute_tig`.

    Parameters
    ----------
    g: Filter
        The filter or filter bank.
    kwargs: dict
        Additional parameters to be passed to the
        :func:`pygsp.filters.Filter.filter` method.
    """
    tig = compute_tig(g, **kwargs)
    return np.linalg.norm(tig, axis=1, ord=2)