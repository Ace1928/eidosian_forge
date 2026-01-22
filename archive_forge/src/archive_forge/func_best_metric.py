from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import ModuleList
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_single_or_multi_val
from torchmetrics.utilities.prints import rank_zero_warn
def best_metric(self, return_step: bool=False) -> Union[None, float, Tuple[float, int], Tuple[None, None], Dict[str, Union[float, None]], Tuple[Dict[str, Union[float, None]], Dict[str, Union[int, None]]]]:
    """Return the highest metric out of all tracked.

        Args:
            return_step: If ``True`` will also return the step with the highest metric value.

        Returns:
            Either a single value or a tuple, depends on the value of ``return_step`` and the object being tracked.

            - If a single metric is being tracked and ``return_step=False`` then a single tensor will be returned
            - If a single metric is being tracked and ``return_step=True`` then a 2-element tuple will be returned,
              where the first value is optimal value and second value is the corresponding optimal step
            - If a metric collection is being tracked and ``return_step=False`` then a single dict will be returned,
              where keys correspond to the different values of the collection and the values are the optimal metric
              value
            - If a metric collection is being bracked and ``return_step=True`` then a 2-element tuple will be returned
              where each is a dict, with keys corresponding to the different values of th collection and the values
              of the first dict being the optimal values and the values of the second dict being the optimal step

            In addition the value in all cases may be ``None`` if the underlying metric does have a proper defined way
            of being optimal or in the case where a nested structure of metrics are being tracked.

        """
    res = self.compute_all()
    if isinstance(res, list):
        rank_zero_warn('Encountered nested structure. You are probably using a metric collection inside a metric collection, or a metric wrapper inside a metric collection, which is not supported by `.best_metric()` method. Returning `None` instead.')
        if return_step:
            return (None, None)
        return None
    if isinstance(self._base_metric, Metric):
        fn = torch.max if self.maximize else torch.min
        try:
            value, idx = fn(res, 0)
            if return_step:
                return (value.item(), idx.item())
            return value.item()
        except (ValueError, RuntimeError) as error:
            rank_zero_warn(f"Encountered the following error when trying to get the best metric: {error}this is probably due to the 'best' not being defined for this metric.Returning `None` instead.", UserWarning)
            if return_step:
                return (None, None)
            return None
    else:
        maximize = self.maximize if isinstance(self.maximize, list) else len(res) * [self.maximize]
        value, idx = ({}, {})
        for i, (k, v) in enumerate(res.items()):
            try:
                fn = torch.max if maximize[i] else torch.min
                out = fn(v, 0)
                value[k], idx[k] = (out[0].item(), out[1].item())
            except (ValueError, RuntimeError) as error:
                rank_zero_warn(f"Encountered the following error when trying to get the best metric for metric {k}:{error} this is probably due to the 'best' not being defined for this metric.Returning `None` instead.", UserWarning)
                value[k], idx[k] = (None, None)
        if return_step:
            return (value, idx)
        return value