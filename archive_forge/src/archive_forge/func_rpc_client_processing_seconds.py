import queue
import socket
import threading
import time
from oslo_config import cfg
from oslo_log import log as logging
from oslo_metrics import message_type
from oslo_utils import eventletutils
from oslo_utils import importutils
def rpc_client_processing_seconds(self, target, method, call_type, duration, timeout=None):
    self.put_rpc_client_metrics_to_txqueue('rpc_client_processing_seconds', message_type.MetricAction('observe', duration), target, method, call_type, timeout)