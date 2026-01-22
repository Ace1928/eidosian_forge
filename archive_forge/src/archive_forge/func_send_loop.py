import queue
import socket
import threading
import time
from oslo_config import cfg
from oslo_log import log as logging
from oslo_metrics import message_type
from oslo_utils import eventletutils
from oslo_utils import importutils
def send_loop(self):
    timeout = self.conf.metrics_thread_stop_timeout
    stoptime = time.time() + timeout
    while stoptime > time.time():
        if self.next_send_metric is None:
            try:
                self.next_send_metric = self.tx_queue.get(timeout=timeout)
            except queue.Empty:
                continue
        try:
            self.send_metric(self.next_send_metric)
            self.next_send_metric = None
            stoptime = time.time() + timeout
        except Exception as e:
            LOG.error('Failed to send metrics: %s. Wait 1 seconds for next try.' % e)
            time.sleep(1)