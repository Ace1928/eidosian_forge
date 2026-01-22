import warnings
from collections.abc import Sequence
import numpy as np
import packaging
import pandas as pd
import scipy
from scipy import stats
from ..data import convert_to_dataset
from ..utils import Numba, _numba_var, _stack, _var_names
from .density_utils import histogram as _histogram
from .stats_utils import _circular_standard_deviation, _sqrt
from .stats_utils import autocov as _autocov
from .stats_utils import not_valid as _not_valid
from .stats_utils import quantile as _quantile
from .stats_utils import stats_variance_2d as svar
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc
def bfmi(data):
    """Calculate the estimated Bayesian fraction of missing information (BFMI).

    BFMI quantifies how well momentum resampling matches the marginal energy distribution. For more
    information on BFMI, see https://arxiv.org/pdf/1604.00695v1.pdf. The current advice is that
    values smaller than 0.3 indicate poor sampling. However, this threshold is
    provisional and may change. See
    `pystan_workflow <http://mc-stan.org/users/documentation/case-studies/pystan_workflow.html>`_
    for more information.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
        If InferenceData, energy variable needs to be found.

    Returns
    -------
    z : array
        The Bayesian fraction of missing information of the model and trace. One element per
        chain in the trace.

    See Also
    --------
    plot_energy : Plot energy transition distribution and marginal energy
                  distribution in HMC algorithms.

    Examples
    --------
    Compute the BFMI of an InferenceData object

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data('radon')
           ...: az.bfmi(data)

    """
    if isinstance(data, np.ndarray):
        return _bfmi(data)
    dataset = convert_to_dataset(data, group='sample_stats')
    if not hasattr(dataset, 'energy'):
        raise TypeError('Energy variable was not found.')
    return _bfmi(dataset.energy.transpose('chain', 'draw'))