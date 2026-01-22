from . import plot
from . import select
from . import utils
from ._lazyload import matplotlib
from scipy import sparse
from scipy import stats
from sklearn import metrics
from sklearn import neighbors
import joblib
import numbers
import numpy as np
import pandas as pd
import warnings
def differential_expression(X, Y, measure='difference', direction='both', gene_names=None, n_jobs=-2):
    """Calculate the most significant genes between two datasets.

    If using ``measure="emd"``, the test statistic is multiplied by the sign of
    the mean differencein order to allow for distinguishing between positive
    and negative shifts. To ignore this, use ``direction="both"`` to sort by the
    absolute value.

    Parameters
    ----------
    X : array-like, shape=[n_cells, n_genes]
    Y : array-like, shape=[m_cells, n_genes]
    measure : {'difference', 'emd', 'ttest', 'ranksum'},
              optional (default: 'difference')
        The measurement to be used to rank genes.
        'difference' is the mean difference between genes.
        'emd' refers to Earth Mover's Distance.
        'ttest' refers to Welch's t-statistic.
        'ranksum' refers to the Wilcoxon rank sum statistic (or the Mann-Whitney
         U statistic).
    direction : {'up', 'down', 'both'}, optional (default: 'both')
        The direction in which to consider genes significant. If 'up', rank
        genes where X > Y. If 'down', rank genes where X < Y. If 'both', rank genes by
        absolute value.
    gene_names : list-like or `None`, optional (default: `None`)
        List of gene names associated with the columns of X and Y
    n_jobs : int, optional (default: -2)
        Number of threads to use if the measurement is parallelizable (currently used
        for EMD).
        If negative, -1 refers to all available cores.

    Returns
    -------
    result : pd.DataFrame
        Ordered DataFrame with a column "gene" and a column named `measure`.
    """
    if direction not in ['up', 'down', 'both']:
        raise ValueError("Expected `direction` in ['up', 'down', 'both']. Got {}".format(direction))
    if measure not in ['difference', 'emd', 'ttest', 'ranksum']:
        raise ValueError("Expected `measure` in ['difference', 'emd', 'ttest', 'ranksum']. Got {}".format(measure))
    if not (len(X.shape) == 2 and len(Y.shape) == 2):
        raise ValueError('Expected `X` and `Y` to be matrices. Got shapes {}, {}'.format(X.shape, Y.shape))
    [X, Y] = utils.check_consistent_columns([X, Y])
    if gene_names is not None:
        if isinstance(X, pd.DataFrame):
            X = select.select_cols(X, idx=gene_names)
            gene_names = X.columns
        if isinstance(Y, pd.DataFrame):
            Y = select.select_cols(Y, idx=gene_names)
            gene_names = Y.columns
        if not len(gene_names) == X.shape[1]:
            raise ValueError('Expected gene_names to have length {}. Got {}'.format(X.shape[1], len(gene_names)))
    elif isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
        gene_names = X.columns
    else:
        gene_names = np.arange(X.shape[1])
    X = utils.to_array_or_spmatrix(X)
    Y = utils.to_array_or_spmatrix(Y)
    if sparse.issparse(X):
        X = X.tocsr()
    if sparse.issparse(Y):
        Y = Y.tocsr()
    if measure == 'difference':
        difference = mean_difference(X, Y)
    if measure == 'ttest':
        difference = t_statistic(X, Y)
    if measure == 'ranksum':
        difference = rank_sum_statistic(X, Y)
    elif measure == 'emd':
        difference = joblib.Parallel(n_jobs)((joblib.delayed(EMD)(select.select_cols(X, idx=i), select.select_cols(Y, idx=i)) for i in range(X.shape[1])))
        difference = np.array(difference) * np.sign(mean_difference(X, Y))
    result = pd.DataFrame({measure: difference}, index=gene_names)
    if direction == 'up':
        if measure == 'ranksum':
            result = result.sort_index().sort_values([measure], ascending=True)
        else:
            result = result.sort_index().sort_values([measure], ascending=False)
    elif direction == 'down':
        if measure == 'ranksum':
            result = result.sort_index().sort_values([measure], ascending=False)
        else:
            result = result.sort_index().sort_values([measure], ascending=True)
    elif direction == 'both':
        result['measure_abs'] = np.abs(difference)
        result = result.sort_index().sort_values(['measure_abs'], ascending=False)
        del result['measure_abs']
    result['rank'] = np.arange(result.shape[0])
    return result