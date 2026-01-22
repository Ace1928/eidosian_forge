from __future__ import annotations
import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable
def marshall_metric_proto(self, metric: MetricProto) -> None:
    """Fill an OpenMetrics `Metric` protobuf object."""
    label = metric.labels.add()
    label.name = 'cache_type'
    label.value = self.category_name
    label = metric.labels.add()
    label.name = 'cache'
    label.value = self.cache_name
    metric_point = metric.metric_points.add()
    metric_point.gauge_value.int_value = self.byte_length