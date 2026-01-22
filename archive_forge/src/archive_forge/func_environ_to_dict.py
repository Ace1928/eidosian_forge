import logging
import os.path
import sys
import traceback as tb
from oslo_config import cfg
from oslo_middleware import base
import webob.dec
import oslo_messaging
from oslo_messaging import notify
@staticmethod
def environ_to_dict(environ):
    """Following PEP 333, server variables are lower case, so don't
        include them.

        """
    return dict(((k, v) for k, v in environ.items() if k.isupper() and k != 'HTTP_X_AUTH_TOKEN'))