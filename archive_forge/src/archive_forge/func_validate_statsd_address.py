import argparse
import copy
import grp
import inspect
import os
import pwd
import re
import shlex
import ssl
import sys
import textwrap
from gunicorn import __version__, util
from gunicorn.errors import ConfigError
from gunicorn.reloader import reloader_engines
def validate_statsd_address(val):
    val = validate_string(val)
    if val is None:
        return None
    unix_hostname_regression = re.match('^unix:(\\d+)$', val)
    if unix_hostname_regression:
        return ('unix', int(unix_hostname_regression.group(1)))
    try:
        address = util.parse_address(val, default_port='8125')
    except RuntimeError:
        raise TypeError("Value must be one of ('host:port', 'unix://PATH')")
    return address