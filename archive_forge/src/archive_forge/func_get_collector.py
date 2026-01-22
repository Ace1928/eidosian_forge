import queue
import socket
import threading
import time
from oslo_config import cfg
from oslo_log import log as logging
from oslo_metrics import message_type
from oslo_utils import eventletutils
from oslo_utils import importutils
def get_collector(conf, metrics_type, **kwargs):
    global threading
    threading = stdlib_threading
    global METRICS_COLLECTOR
    if METRICS_COLLECTOR is None:
        METRICS_COLLECTOR = MetricsCollectorClient(conf, metrics_type, **kwargs)
    return METRICS_COLLECTOR