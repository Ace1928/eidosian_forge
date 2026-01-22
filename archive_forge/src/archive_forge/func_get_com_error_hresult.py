import ctypes
import inspect
from pkg_resources import parse_version
import textwrap
import time
import types
import eventlet
from eventlet import tpool
import netaddr
from oslo_concurrency import lockutils
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
import six
from os_win import constants
from os_win import exceptions
def get_com_error_hresult(com_error):
    try:
        return ctypes.c_uint(com_error.excepinfo[5]).value
    except Exception:
        LOG.debug('Unable to retrieve COM error hresult: %s', com_error)