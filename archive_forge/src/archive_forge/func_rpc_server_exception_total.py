import queue
import socket
import threading
import time
from oslo_config import cfg
from oslo_log import log as logging
from oslo_metrics import message_type
from oslo_utils import eventletutils
from oslo_utils import importutils
def rpc_server_exception_total(self, target, endpoint, ns, ver, method, exception):
    self.put_rpc_server_metrics_to_txqueue('rpc_server_exception_total', message_type.MetricAction('inc', None), target, endpoint, ns, ver, method, exception=exception)