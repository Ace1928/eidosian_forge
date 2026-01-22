import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
def new_stats_exporter(option):
    """new_stats_exporter returns an exporter
    that exports stats to Prometheus.
    """
    if option.namespace == '':
        raise ValueError('Namespace can not be empty string.')
    collector = new_collector(option)
    exporter = PrometheusStatsExporter(options=option, gatherer=option.registry, collector=collector)
    return exporter