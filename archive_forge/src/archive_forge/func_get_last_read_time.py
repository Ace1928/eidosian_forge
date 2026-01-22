import functools
import signal
import time
from oslo_utils import importutils
from osprofiler.drivers import base
def get_last_read_time(self):
    return self.last_read_time