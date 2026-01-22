import platform as pf
from typing import Any, Iterable, Optional
from .metrics_core import GaugeMetricFamily, Metric
from .registry import Collector, CollectorRegistry, REGISTRY
Collector for python platform information