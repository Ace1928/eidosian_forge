import json
import os
import re
import sys
from importlib.util import find_spec
import pydevd
from _pydevd_bundle import pydevd_runpy as runpy
import debugpy
from debugpy.common import log
from debugpy.server import api
import codecs;
import json;
import sys;
import attach_pid_injected;
def set_address(mode):

    def do(arg, it):
        if options.address is not None:
            raise ValueError('--listen and --connect are mutually exclusive')
        value = next(it)
        host, sep, port = value.partition(':')
        if not sep:
            host = '127.0.0.1'
            port = value
        try:
            port = int(port)
        except Exception:
            port = -1
        if not 0 <= port < 2 ** 16:
            raise ValueError('invalid port number')
        options.mode = mode
        options.address = (host, port)
    return do