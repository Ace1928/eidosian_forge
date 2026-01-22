import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg
from pygsp import graphs, filters, utils
def pyramid_analysis(Gs, f, **kwargs):
    """Compute the graph pyramid transform coefficients.

    Parameters
    ----------
    Gs : list of graphs
        A multiresolution sequence of graph structures.
    f : ndarray
        Graph signal to analyze.
    h_filters : list
        A list of filter that will be used for the analysis and sythesis operator.
        If only one filter is given, it will be used for all levels.
        Default is h(x) = 1 / (2x+1)

    Returns
    -------
    ca : ndarray
        Coarse approximation at each level
    pe : ndarray
        Prediction error at each level
    h_filters : list
        Graph spectral filters applied

    References
    ----------
    See :cite:`shuman2013framework` and :cite:`pesenson2009variational`.

    """
    if np.shape(f)[0] != Gs[0].N:
        raise ValueError('PYRAMID ANALYSIS: The signal to analyze should have the same dimension as the first graph.')
    levels = len(Gs) - 1
    h_filters = kwargs.pop('h_filters', lambda x: 1.0 / (2 * x + 1))
    if not isinstance(h_filters, list):
        if hasattr(h_filters, '__call__'):
            logger.warning('Converting filters into a list.')
            h_filters = [h_filters]
        else:
            logger.error('Filters must be a list of functions.')
    if len(h_filters) == 1:
        h_filters = h_filters * levels
    elif len(h_filters) != levels:
        message = 'The number of filters must be one or equal to {}.'.format(levels)
        raise ValueError(message)
    ca = [f]
    pe = []
    for i in range(levels):
        s_low = _analysis(filters.Filter(Gs[i], h_filters[i]), ca[i], **kwargs)
        ca.append(s_low[Gs[i + 1].mr['idx']])
        s_pred = interpolate(Gs[i], ca[i + 1], Gs[i + 1].mr['idx'], **kwargs)
        pe.append(ca[i] - s_pred)
    return (ca, pe)