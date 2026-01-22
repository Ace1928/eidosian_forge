import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def plot_missing_pattern(self, ax=None, row_order='pattern', column_order='pattern', hide_complete_rows=False, hide_complete_columns=False, color_row_patterns=True):
    """
        Generate an image showing the missing data pattern.

        Parameters
        ----------
        ax : AxesSubplot
            Axes on which to draw the plot.
        row_order : str
            The method for ordering the rows.  Must be one of 'pattern',
            'proportion', or 'raw'.
        column_order : str
            The method for ordering the columns.  Must be one of 'pattern',
            'proportion', or 'raw'.
        hide_complete_rows : bool
            If True, rows with no missing values are not drawn.
        hide_complete_columns : bool
            If True, columns with no missing values are not drawn.
        color_row_patterns : bool
            If True, color the unique row patterns, otherwise use grey
            and white as colors.

        Returns
        -------
        A figure containing a plot of the missing data pattern.
        """
    miss = np.zeros(self.data.shape)
    cols = self.data.columns
    for j, col in enumerate(cols):
        ix = self.ix_miss[col]
        miss[ix, j] = 1
    if column_order == 'proportion':
        ix = np.argsort(miss.mean(0))
    elif column_order == 'pattern':
        cv = np.cov(miss.T)
        u, s, vt = np.linalg.svd(cv, 0)
        ix = np.argsort(cv[:, 0])
    elif column_order == 'raw':
        ix = np.arange(len(cols))
    else:
        raise ValueError(column_order + ' is not an allowed value for `column_order`.')
    miss = miss[:, ix]
    cols = [cols[i] for i in ix]
    if row_order == 'proportion':
        ix = np.argsort(miss.mean(1))
    elif row_order == 'pattern':
        x = 2 ** np.arange(miss.shape[1])
        rky = np.dot(miss, x)
        ix = np.argsort(rky)
    elif row_order == 'raw':
        ix = np.arange(miss.shape[0])
    else:
        raise ValueError(row_order + ' is not an allowed value for `row_order`.')
    miss = miss[ix, :]
    if hide_complete_rows:
        ix = np.flatnonzero((miss == 1).any(1))
        miss = miss[ix, :]
    if hide_complete_columns:
        ix = np.flatnonzero((miss == 1).any(0))
        miss = miss[:, ix]
        cols = [cols[i] for i in ix]
    from statsmodels.graphics import utils as gutils
    from matplotlib.colors import LinearSegmentedColormap
    if ax is None:
        fig, ax = gutils.create_mpl_ax(ax)
    else:
        fig = ax.get_figure()
    if color_row_patterns:
        x = 2 ** np.arange(miss.shape[1])
        rky = np.dot(miss, x)
        _, rcol = np.unique(rky, return_inverse=True)
        miss *= 1 + rcol[:, None]
        ax.imshow(miss, aspect='auto', interpolation='nearest', cmap='gist_ncar_r')
    else:
        cmap = LinearSegmentedColormap.from_list('_', ['white', 'darkgrey'])
        ax.imshow(miss, aspect='auto', interpolation='nearest', cmap=cmap)
    ax.set_ylabel('Cases')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    return fig