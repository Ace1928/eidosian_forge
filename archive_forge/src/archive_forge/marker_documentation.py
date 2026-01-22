from .. import select
from .. import stats
from .. import utils
from .tools import label_axis
from .utils import _get_figure
from .utils import parse_fontsize
from .utils import shift_ticklabels
from .utils import show
from .utils import temp_fontsize
from scipy.cluster import hierarchy
import numpy as np
import pandas as pd
Plot marker gene enrichment.

    Generate a plot indicating the expression level and enrichment of
    a set of marker genes for each cluster.

    Color of each point indicates the expression of each gene in each cluster.
    The size of each point indicates how differentially expressed each gene is
    in each cluster.

    Parameters
    ----------
    data : array-like, shape=[n_cells, n_genes]
        Gene expression data for calculating expression statistics.
    clusters : list-like, shape=[n_cells]
        Cluster assignments for each cell. Should be ints
        like the output of most sklearn.cluster methods.
    markers : dict or list-like
        If a dictionary, keys represent tissues and
        values being a list of marker genes in each tissue.
        If a list, a list of marker genes.
    gene_names : list-like, shape=[n_genes]
        List of gene names.
    normalize_{expression,emd} : bool, optional (default: True)
        Normalize the expression and EMD of each row.
    reorder_{tissues,markers} : bool, optional (default: True)
        Reorder tissues and markers according to hierarchical clustering=
    cmap : str or matplotlib colormap, optional (default: 'inferno')
        Colormap with which to color points.
    title : str or None, optional (default: None)
        Title for the plot
    figsize : tuple or None, optional (default: None)
        If not None, sets the figure size (width, height)
    ax : `matplotlib.Axes` or None, optional (default: None)
        Axis to plot on. If None, a new axis will be created.
    fontsize : int or None, optional (default: None)
        Base fontsize.

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn

    Example
    -------
    >>> markers = {'Adaxial - Immature': ['myl10', 'myod1'],
                   'Adaxial - Mature': ['myog'],
                   'Presomitic mesoderm': ['tbx6', 'msgn1', 'tbx16'],
                   'Forming somites': ['mespba', 'ripply2'],
                   'Somites': ['meox1', 'ripply1', 'aldh1a2']}
    >>> cluster_marker_plot(data, clusters, gene_names, markers,
                            title="Tailbud - PSM")
    