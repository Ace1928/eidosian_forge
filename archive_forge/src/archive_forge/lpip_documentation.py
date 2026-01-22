from typing import Any, ClassVar, List, Optional, Sequence, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.image.lpips import _LPIPS, _lpips_compute, _lpips_update, _NoTrainLpips
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCHVISION_AVAILABLE
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
            >>> from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            >>> metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
            >>> metric.update(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            >>> metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100)))
            >>> fig_, ax_ = metric.plot(values)

        