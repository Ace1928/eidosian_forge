import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
def clustermap(data, *, pivot_kws=None, method='average', metric='euclidean', z_score=None, standard_scale=None, figsize=(10, 10), cbar_kws=None, row_cluster=True, col_cluster=True, row_linkage=None, col_linkage=None, row_colors=None, col_colors=None, mask=None, dendrogram_ratio=0.2, colors_ratio=0.03, cbar_pos=(0.02, 0.8, 0.05, 0.18), tree_kws=None, **kwargs):
    """
    Plot a matrix dataset as a hierarchically-clustered heatmap.

    This function requires scipy to be available.

    Parameters
    ----------
    data : 2D array-like
        Rectangular data for clustering. Cannot contain NAs.
    pivot_kws : dict, optional
        If `data` is a tidy dataframe, can provide keyword arguments for
        pivot to create a rectangular dataframe.
    method : str, optional
        Linkage method to use for calculating clusters. See
        :func:`scipy.cluster.hierarchy.linkage` documentation for more
        information.
    metric : str, optional
        Distance metric to use for the data. See
        :func:`scipy.spatial.distance.pdist` documentation for more options.
        To use different metrics (or methods) for rows and columns, you may
        construct each linkage matrix yourself and provide them as
        `{row,col}_linkage`.
    z_score : int or None, optional
        Either 0 (rows) or 1 (columns). Whether or not to calculate z-scores
        for the rows or the columns. Z scores are: z = (x - mean)/std, so
        values in each row (column) will get the mean of the row (column)
        subtracted, then divided by the standard deviation of the row (column).
        This ensures that each row (column) has mean of 0 and variance of 1.
    standard_scale : int or None, optional
        Either 0 (rows) or 1 (columns). Whether or not to standardize that
        dimension, meaning for each row or column, subtract the minimum and
        divide each by its maximum.
    figsize : tuple of (width, height), optional
        Overall size of the figure.
    cbar_kws : dict, optional
        Keyword arguments to pass to `cbar_kws` in :func:`heatmap`, e.g. to
        add a label to the colorbar.
    {row,col}_cluster : bool, optional
        If ``True``, cluster the {rows, columns}.
    {row,col}_linkage : :class:`numpy.ndarray`, optional
        Precomputed linkage matrix for the rows or columns. See
        :func:`scipy.cluster.hierarchy.linkage` for specific formats.
    {row,col}_colors : list-like or pandas DataFrame/Series, optional
        List of colors to label for either the rows or columns. Useful to evaluate
        whether samples within a group are clustered together. Can use nested lists or
        DataFrame for multiple color levels of labeling. If given as a
        :class:`pandas.DataFrame` or :class:`pandas.Series`, labels for the colors are
        extracted from the DataFrames column names or from the name of the Series.
        DataFrame/Series colors are also matched to the data by their index, ensuring
        colors are drawn in the correct order.
    mask : bool array or DataFrame, optional
        If passed, data will not be shown in cells where `mask` is True.
        Cells with missing values are automatically masked. Only used for
        visualizing, not for calculating.
    {dendrogram,colors}_ratio : float, or pair of floats, optional
        Proportion of the figure size devoted to the two marginal elements. If
        a pair is given, they correspond to (row, col) ratios.
    cbar_pos : tuple of (left, bottom, width, height), optional
        Position of the colorbar axes in the figure. Setting to ``None`` will
        disable the colorbar.
    tree_kws : dict, optional
        Parameters for the :class:`matplotlib.collections.LineCollection`
        that is used to plot the lines of the dendrogram tree.
    kwargs : other keyword arguments
        All other keyword arguments are passed to :func:`heatmap`.

    Returns
    -------
    :class:`ClusterGrid`
        A :class:`ClusterGrid` instance.

    See Also
    --------
    heatmap : Plot rectangular data as a color-encoded matrix.

    Notes
    -----
    The returned object has a ``savefig`` method that should be used if you
    want to save the figure object without clipping the dendrograms.

    To access the reordered row indices, use:
    ``clustergrid.dendrogram_row.reordered_ind``

    Column indices, use:
    ``clustergrid.dendrogram_col.reordered_ind``

    Examples
    --------

    .. include:: ../docstrings/clustermap.rst

    """
    if _no_scipy:
        raise RuntimeError('clustermap requires scipy to be available')
    plotter = ClusterGrid(data, pivot_kws=pivot_kws, figsize=figsize, row_colors=row_colors, col_colors=col_colors, z_score=z_score, standard_scale=standard_scale, mask=mask, dendrogram_ratio=dendrogram_ratio, colors_ratio=colors_ratio, cbar_pos=cbar_pos)
    return plotter.plot(metric=metric, method=method, colorbar_kws=cbar_kws, row_cluster=row_cluster, col_cluster=col_cluster, row_linkage=row_linkage, col_linkage=col_linkage, tree_kws=tree_kws, **kwargs)