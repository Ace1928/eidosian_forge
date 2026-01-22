from __future__ import (absolute_import, division, print_function)
import atexit
import datetime
import itertools
import json
import os
import re
import ssl
import sys
import uuid
from time import time
from jinja2 import Environment
from ansible.module_utils.six import integer_types, PY3
from ansible.module_utils.six.moves import configparser
def select_chain_match(inlist, key, pattern):
    """Get a key from a list of dicts, squash values to a single list, then filter"""
    outlist = [x[key] for x in inlist]
    outlist = list(itertools.chain(*outlist))
    outlist = [x for x in outlist if regex_match(x, pattern)]
    return outlist