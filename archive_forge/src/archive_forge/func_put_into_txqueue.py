import queue
import socket
import threading
import time
from oslo_config import cfg
from oslo_log import log as logging
from oslo_metrics import message_type
from oslo_utils import eventletutils
from oslo_utils import importutils
def put_into_txqueue(self, metrics_name, action, **labels):
    labels['process'] = self.conf.metrics_process_name
    m = message_type.Metric('oslo_messaging', metrics_name, action, **labels)
    try:
        self.tx_queue.put_nowait(m)
    except queue.Full:
        LOG.warning('tx queues is already full(%s/%s). Fails to send the metrics(%s)' % (self.tx_queue.qsize(), self.tx_queue.maxsize, m))
    if not self.send_thread.is_alive():
        self.send_thread = threading.Thread(target=self.send_loop)
        self.send_thread.start()