from typing import Any, Dict, Optional, Sequence, Union
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.abstract import WrapperMetric
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
            >>> from torchmetrics.wrappers import MinMaxMetric
            >>> from torchmetrics.classification import BinaryAccuracy
            >>> metric = MinMaxMetric(BinaryAccuracy())
            >>> metric.update(torch.randint(2, (20,)), torch.randint(2, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.wrappers import MinMaxMetric
            >>> from torchmetrics.classification import BinaryAccuracy
            >>> metric = MinMaxMetric(BinaryAccuracy())
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.randint(2, (20,)), torch.randint(2, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        