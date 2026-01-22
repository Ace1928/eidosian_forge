import collections
import contextlib
import copy
import gc
import itertools
import os
import random
import threading
from absl import logging
import numpy as np
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import execute
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def configure_coordination_service(self, service_type, service_leader='', enable_health_check=True, cluster_register_timeout_in_ms=0, heartbeat_timeout_in_ms=0, shutdown_barrier_timeout_in_ms=0, coordinated_jobs=None, allow_new_incarnation_to_reconnect=False):
    """Enable distributed coordination service with specified configs."""
    if self._context_handle:
        logging.warning('Configuring coordination service type may not be effective because the context is already initialized.')
    config = coordination_config_pb2.CoordinationServiceConfig()
    config.service_type = service_type
    if service_leader:
        config.service_leader = pydev.canonical_name(service_leader)
    config.enable_health_check = enable_health_check
    config.cluster_register_timeout_in_ms = cluster_register_timeout_in_ms
    config.heartbeat_timeout_in_ms = heartbeat_timeout_in_ms
    config.shutdown_barrier_timeout_in_ms = shutdown_barrier_timeout_in_ms
    config.allow_new_incarnation_to_reconnect = allow_new_incarnation_to_reconnect
    if coordinated_jobs is not None:
        if isinstance(coordinated_jobs, list):
            config.coordinated_job_list.extend(coordinated_jobs)
        else:
            raise ValueError('`coordinated_jobs` must be list[CoordinatedJob] or None, but got: %s' % (coordinated_jobs,))
    self._coordination_service_config = config