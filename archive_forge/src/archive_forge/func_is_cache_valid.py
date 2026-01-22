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
def is_cache_valid(self):
    """ Determines if the cache files have expired, or if it is still valid """
    valid = False
    if os.path.isfile(self.cache_path_cache):
        mod_time = os.path.getmtime(self.cache_path_cache)
        current_time = time()
        if mod_time + self.cache_max_age > current_time:
            valid = True
    return valid