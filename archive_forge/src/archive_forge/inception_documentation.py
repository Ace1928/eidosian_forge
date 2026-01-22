from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCH_FIDELITY_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.image.inception import InceptionScore
            >>> metric = InceptionScore()
            >>> metric.update(torch.randint(0, 255, (50, 3, 299, 299), dtype=torch.uint8))
            >>> fig_, ax_ = metric.plot()  # the returned plot only shows the mean value by default

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.inception import InceptionScore
            >>> metric = InceptionScore()
            >>> values = [ ]
            >>> for _ in range(3):
            ...     # we index by 0 such that only the mean value is plotted
            ...     values.append(metric(torch.randint(0, 255, (50, 3, 299, 299), dtype=torch.uint8))[0])
            >>> fig_, ax_ = metric.plot(values)

        