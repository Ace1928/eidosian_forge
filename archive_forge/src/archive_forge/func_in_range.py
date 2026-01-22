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
def in_range(parser, start, stop):

    def parse(s):
        n = parser(s)
        if start is not None and n < start:
            raise ValueError('must be >= {0}'.format(start))
        if stop is not None and n >= stop:
            raise ValueError('must be < {0}'.format(stop))
        return n
    return parse