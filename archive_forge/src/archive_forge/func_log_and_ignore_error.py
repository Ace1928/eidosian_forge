import logging
import os.path
import sys
import traceback as tb
from oslo_config import cfg
from oslo_middleware import base
import webob.dec
import oslo_messaging
from oslo_messaging import notify
def log_and_ignore_error(fn):

    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            LOG.exception('An exception occurred processing the API call: %s ', e)
    return wrapped