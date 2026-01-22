import json
import logging
import os
import re
import threading
import time
import traceback
from collections import namedtuple
from typing import List, Tuple, Any, Dict
from prometheus_client.core import (
from opencensus.metrics.export.value import ValueDouble
from opencensus.stats import aggregation
from opencensus.stats import measure as measure_module
from opencensus.stats.view_manager import ViewManager
from opencensus.stats.stats_recorder import StatsRecorder
from opencensus.stats.base_exporter import StatsExporter
from prometheus_client.core import Metric as PrometheusMetric
from opencensus.stats.aggregation_data import (
from opencensus.stats.view import View
from opencensus.tags import tag_key as tag_key_module
from opencensus.tags import tag_map as tag_map_module
from opencensus.tags import tag_value as tag_value_module
import ray
from ray._raylet import GcsClient
from ray.core.generated.metrics_pb2 import Metric
def fix_grpc_metric(metric: Metric):
    """
    Fix the inbound `opencensus.proto.metrics.v1.Metric` protos to make it acceptable
    by opencensus.stats.DistributionAggregationData.

    - metric name: gRPC OpenCensus metrics have names with slashes and dots, e.g.
    `grpc.io/client/server_latency`[1]. However Prometheus metric names only take
    alphanums,underscores and colons[2]. We santinize the name by replacing non-alphanum
    chars to underscore, like the official opencensus prometheus exporter[3].
    - distribution bucket bounds: The Metric proto asks distribution bucket bounds to
    be > 0 [4]. However, gRPC OpenCensus metrics have their first bucket bound == 0 [1].
    This makes the `DistributionAggregationData` constructor to raise Exceptions. This
    applies to all bytes and milliseconds (latencies). The fix: we update the initial 0
    bounds to be 0.000_000_1. This will not affect the precision of the metrics, since
    we don't expect any less-than-1 bytes, or less-than-1-nanosecond times.

    [1] https://github.com/census-instrumentation/opencensus-specs/blob/master/stats/gRPC.md#units  # noqa: E501
    [2] https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
    [3] https://github.com/census-instrumentation/opencensus-cpp/blob/50eb5de762e5f87e206c011a4f930adb1a1775b1/opencensus/exporters/stats/prometheus/internal/prometheus_utils.cc#L39 # noqa: E501
    [4] https://github.com/census-instrumentation/opencensus-proto/blob/master/src/opencensus/proto/metrics/v1/metrics.proto#L218 # noqa: E501
    """
    if not metric.metric_descriptor.name.startswith('grpc.io/'):
        return
    metric.metric_descriptor.name = RE_NON_ALPHANUMS.sub('_', metric.metric_descriptor.name)
    for series in metric.timeseries:
        for point in series.points:
            if point.HasField('distribution_value'):
                dist_value = point.distribution_value
                bucket_bounds = dist_value.bucket_options.explicit.bounds
                if len(bucket_bounds) > 0 and bucket_bounds[0] == 0:
                    bucket_bounds[0] = 1e-07