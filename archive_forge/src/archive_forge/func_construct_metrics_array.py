import time
import logging
import typing as tp
from IPython.display import display
from copy import deepcopy
from typing import List, Optional, Any, Union
from .ipythonwidget import MetricWidget
@staticmethod
def construct_metrics_array(metrics_positions: tp.Dict[str, int], metrics: tp.Dict[str, float]) -> List[float]:
    array: List[float] = [0.0] * len(metrics_positions)
    assert set(metrics.keys()) == set(metrics_positions.keys()), f'Not all metrics were passed while logging, expected following: {', '.join(list(metrics_positions.keys()))}'
    for metric, value in metrics.items():
        assert isinstance(value, float), 'Type of metric {metric} should be float'
        array[metrics_positions[metric]] = value
    return array