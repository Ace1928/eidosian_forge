from itertools import product
from math import ceil, floor, sqrt
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union, no_type_check
import numpy as np
import torch
from torch import Tensor
from torchmetrics.utilities.imports import _LATEX_AVAILABLE, _MATPLOTLIB_AVAILABLE, _SCIENCEPLOT_AVAILABLE
@style_change(_style)
def plot_curve(curve: Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]], score: Optional[Tensor]=None, ax: Optional[_AX_TYPE]=None, label_names: Optional[Tuple[str, str]]=None, legend_name: Optional[str]=None, name: Optional[str]=None) -> _PLOT_OUT_TYPE:
    """Inspired by: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_plot/roc_curve.py.

    Plots a curve object

    Args:
        curve: a tuple of (x, y, t) where x and y are the coordinates of the curve and t are the thresholds used
            to compute the curve
        score: optional area under the curve added as label to the plot
        ax: Axis from a figure
        label_names: Tuple containing the names of the x and y axis
        legend_name: Name of the curve to be used in the legend
        name: Custom name to describe the metric

    Returns:
        A tuple consisting of the figure and respective ax objects (or array of ax objects) of the generated figure

    Raises:
        ModuleNotFoundError:
            If `matplotlib` is not installed
        ValueError:
            If `curve` does not have 3 elements, being in the wrong format
    """
    if len(curve) < 2:
        raise ValueError('Expected 2 or 3 elements in curve but got {len(curve)}')
    x, y = curve[:2]
    _error_on_missing_matplotlib()
    fig, ax = plt.subplots() if ax is None else (None, ax)
    if isinstance(x, Tensor) and isinstance(y, Tensor) and (x.ndim == 1) and (y.ndim == 1):
        label = f'AUC={score.item():0.3f}' if score is not None else None
        ax.plot(x.detach().cpu(), y.detach().cpu(), linestyle='-', linewidth=2, label=label)
        if label_names is not None:
            ax.set_xlabel(label_names[0])
            ax.set_ylabel(label_names[1])
        if label is not None:
            ax.legend()
    elif isinstance(x, list) and isinstance(y, list) or (isinstance(x, Tensor) and isinstance(y, Tensor) and (x.ndim == 2) and (y.ndim == 2)):
        for i, (x_, y_) in enumerate(zip(x, y)):
            label = f'{legend_name}_{i}' if legend_name is not None else str(i)
            label += f' AUC={score[i].item():0.3f}' if score is not None else ''
            ax.plot(x_.detach().cpu(), y_.detach().cpu(), label=label)
            ax.legend()
    else:
        raise ValueError(f'Unknown format for argument `x` and `y`. Expected either list or tensors but got {type(x)} and {type(y)}.')
    ax.grid(True)
    ax.set_title(name)
    return (fig, ax)