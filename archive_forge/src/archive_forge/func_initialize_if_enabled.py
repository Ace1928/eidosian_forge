import errno
import gc
import logging
import os
import pprint
import sys
import tempfile
import traceback
import eventlet.backdoor
import greenlet
import yappi
from eventlet.green import socket
from oslo_service._i18n import _
from oslo_service import _options
def initialize_if_enabled(conf):
    where_running_thread = _initialize_if_enabled(conf)
    if not where_running_thread:
        return None
    else:
        where_running, _thread = where_running_thread
        return where_running