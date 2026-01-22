import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
def log_to_streamhandler(stream=None):
    stream = stream or sys.stderr
    ch = logging.StreamHandler(stream)
    LOG.setLevel(logging.DEBUG)
    LOG.addHandler(ch)