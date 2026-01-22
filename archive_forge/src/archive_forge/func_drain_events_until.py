import socket
import threading
import time
from collections import deque
from queue import Empty
from time import sleep
from weakref import WeakKeyDictionary
from kombu.utils.compat import detect_environment
from celery import states
from celery.exceptions import TimeoutError
from celery.utils.threads import THREAD_TIMEOUT_MAX
def drain_events_until(self, p, timeout=None, on_interval=None):
    return self.drainer.drain_events_until(p, timeout=timeout, on_interval=on_interval)