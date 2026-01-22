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
def facts_from_vobj(self, vobj, level=0):
    """ Traverse a VM object and return a json compliant data structure """
    if level == 0:
        try:
            self.debugl('get facts for %s' % vobj.name)
        except Exception as e:
            self.debugl(e)
    rdata = {}
    methods = dir(vobj)
    methods = [str(x) for x in methods if not x.startswith('_')]
    methods = [x for x in methods if x not in self.bad_types]
    methods = [x for x in methods if not x.lower() in self.skip_keys]
    methods = sorted(methods)
    for method in methods:
        try:
            methodToCall = getattr(vobj, method)
        except Exception as e:
            continue
        if callable(methodToCall):
            continue
        if self.lowerkeys:
            method = method.lower()
        rdata[method] = self._process_object_types(methodToCall, thisvm=vobj, inkey=method)
    return rdata