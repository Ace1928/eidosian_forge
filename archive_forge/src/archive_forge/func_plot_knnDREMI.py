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
@utils._with_pkg(pkg='matplotlib', min_version=3)
def plot_knnDREMI(dremi, mutual_info, x, y, n_bins, n_mesh, density, bin_density, drevi, figsize=(12, 3.5), filename=None, xlabel='Feature 1', ylabel='Feature 2', title_fontsize=18, label_fontsize=16, dpi=150):
    """Plot results of DREMI.

    Create plots of the data like those seen in
    Fig 5C/D of van Dijk et al. 2018. [1]_
    Note that this function is not designed to be called manually. Instead
    create plots by running `scprep.stats.knnDREMI` with `plot=True`.

    Parameters
    ----------
    figsize : tuple, optional (default: (12, 3.5))
        Matplotlib figure size
    filename : str or `None`, optional (default: None)
        If given, saves the results to a file
    xlabel : str, optional (default: "Feature 1")
        The name of the gene shown on the x axis
    ylabel : str, optional (default: "Feature 2")
        The name of the gene shown on the y axis
    title_fontsize : int, optional (default: 18)
        Font size for figure titles
    label_fontsize : int, optional (default: 16)
        Font size for axis labels
    dpi : int, optional (default: 150)
        Dots per inch for saved figure
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    axes[0].scatter(x, y, c='k', s=4)
    axes[0].set_title('Input\ndata', fontsize=title_fontsize)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel(xlabel, fontsize=label_fontsize)
    axes[0].set_ylabel(ylabel, fontsize=label_fontsize)
    n = (n_mesh + 1) * n_bins + 1
    axes[1].imshow(np.log(density.reshape(n, n)), cmap='inferno', origin='lower', aspect='auto')
    for b in np.linspace(0, n, n_bins + 1):
        axes[1].axhline(b - 0.5, c='grey', linewidth=1)
    for b in np.linspace(0, n, n_bins + 1):
        axes[1].axvline(b - 0.5, c='grey', linewidth=1)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title('kNN\nDensity', fontsize=title_fontsize)
    axes[1].set_xlabel(xlabel, fontsize=label_fontsize)
    axes[2].imshow(bin_density, cmap='inferno', origin='lower', aspect='auto')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title('Joint Prob.\nMI={:.2f}'.format(mutual_info), fontsize=title_fontsize)
    axes[2].set_xlabel(xlabel, fontsize=label_fontsize)
    axes[3].imshow(drevi, cmap='inferno', origin='lower', aspect='auto')
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    axes[3].set_title('Conditional Prob.\nDREMI={:.2f}'.format(dremi), fontsize=title_fontsize)
    axes[3].set_xlabel(xlabel, fontsize=label_fontsize)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=dpi)
    plot.utils.show(fig)