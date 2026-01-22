from itertools import product
from math import ceil, floor, sqrt
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union, no_type_check
import numpy as np
import torch
from torch import Tensor
from torchmetrics.utilities.imports import _LATEX_AVAILABLE, _MATPLOTLIB_AVAILABLE, _SCIENCEPLOT_AVAILABLE
@style_change(_style)
@no_type_check
def plot_confusion_matrix(confmat: Tensor, ax: Optional[_AX_TYPE]=None, add_text: bool=True, labels: Optional[List[Union[int, str]]]=None) -> _PLOT_OUT_TYPE:
    """Plot an confusion matrix.

    Inspired by: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_plot/confusion_matrix.py.
    Works for both binary, multiclass and multilabel confusion matrices.

    Args:
        confmat: the confusion matrix. Either should be an [N,N] matrix in the binary and multiclass cases or an
            [N, 2, 2] matrix for multilabel classification
        ax: Axis from a figure. If not provided, a new figure and axis will be created
        add_text: if text should be added to each cell with the given value
        labels: labels to add the x- and y-axis

    Returns:
        A tuple consisting of the figure and respective ax objects (or array of ax objects) of the generated figure

    Raises:
        ModuleNotFoundError:
            If `matplotlib` is not installed

    """
    _error_on_missing_matplotlib()
    if confmat.ndim == 3:
        nb, n_classes = (confmat.shape[0], 2)
        rows, cols = _get_col_row_split(nb)
    else:
        nb, n_classes, rows, cols = (1, confmat.shape[0], 1, 1)
    if labels is not None and confmat.ndim != 3 and (len(labels) != n_classes):
        raise ValueError(f'Expected number of elements in arg `labels` to match number of labels in confmat but got {len(labels)} and {n_classes}')
    if confmat.ndim == 3:
        fig_label = labels or np.arange(nb)
        labels = list(map(str, range(n_classes)))
    else:
        fig_label = None
        labels = labels or np.arange(n_classes).tolist()
    fig, axs = plt.subplots(nrows=rows, ncols=cols, constrained_layout=True) if ax is None else (ax.get_figure(), ax)
    axs = trim_axs(axs, nb)
    for i in range(nb):
        ax = axs[i] if rows != 1 and cols != 1 else axs
        if fig_label is not None:
            ax.set_title(f'Label {fig_label[i]}', fontsize=15)
        ax.imshow(confmat[i].cpu().detach() if confmat.ndim == 3 else confmat.cpu().detach())
        if i // cols == rows - 1:
            ax.set_xlabel('Predicted class', fontsize=15)
        if i % cols == 0:
            ax.set_ylabel('True class', fontsize=15)
        ax.set_xticks(list(range(n_classes)))
        ax.set_yticks(list(range(n_classes)))
        ax.set_xticklabels(labels, rotation=45, fontsize=10)
        ax.set_yticklabels(labels, rotation=25, fontsize=10)
        if add_text:
            for ii, jj in product(range(n_classes), range(n_classes)):
                val = confmat[i, ii, jj] if confmat.ndim == 3 else confmat[ii, jj]
                ax.text(jj, ii, str(round(val.item(), 2)), ha='center', va='center', fontsize=15)
    return (fig, axs)