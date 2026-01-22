import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def locate_encode(name):
    if re.match('decode\\..+', name):
        return 'filters.' + name
    else:
        return filters.DEFAULT_ESCAPES.get(name, name)