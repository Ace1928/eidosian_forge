import logging
import os
import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@staticmethod
def get_service_addr(service_addr: Optional[str]=None, tpu_name: Optional[str]=None, compute_zone: Optional[str]=None, core_project: Optional[str]=None) -> str:
    if service_addr is not None:
        if tpu_name is not None:
            logger.warn('Both service_addr and tpu_name arguments provided. Ignoring tpu_name and using service_addr.')
    else:
        tpu_name = tpu_name or os.environ.get('TPU_NAME')
        if tpu_name is None:
            raise Exception('Required environment variable TPU_NAME.')
        compute_zone = compute_zone or os.environ.get('CLOUDSDK_COMPUTE_ZONE')
        core_project = core_project or os.environ.get('CLOUDSDK_CORE_PROJECT')
        try:
            from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
            service_addr = tpu_cluster_resolver.TPUClusterResolver([tpu_name], zone=compute_zone, project=core_project).get_master()
        except (ValueError, TypeError):
            raise ValueError('Failed to find TPU. Try specifying TPU zone (via CLOUDSDK_COMPUTE_ZONE environment variable) and GCP project (via CLOUDSDK_CORE_PROJECT environment variable).')
    service_addr = service_addr.replace('grpc://', '').replace(':8470', ':8466')
    return service_addr