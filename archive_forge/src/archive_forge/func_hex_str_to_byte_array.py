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
def hex_str_to_byte_array(string):
    string = string.lower().replace('0x', '')
    if len(string) % 2:
        string = '0%s' % string
    return bytearray([int(hex_byte, 16) for hex_byte in textwrap.wrap(string, 2)])