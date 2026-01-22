import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
def split_filter_value_for_quotes(value):
    """Split filter values

    Split values by commas and quotes for 'in' operator, according api-wg.
    """
    validate_quotes(value)
    tmp = re.compile('\n        "(                 # if found a double-quote\n           [^\\"\\\\]*        # take characters either non-quotes or backslashes\n           (?:\\\\.          # take backslashes and character after it\n            [^\\"\\\\]*)*     # take characters either non-quotes or backslashes\n         )                 # before double-quote\n        ",?                # a double-quote with comma maybe\n        | ([^,]+),?        # if not found double-quote take any non-comma\n                           # characters with comma maybe\n        | ,                # if we have only comma take empty string\n        ', re.VERBOSE)
    return [val[0] or val[1] for val in re.findall(tmp, value)]