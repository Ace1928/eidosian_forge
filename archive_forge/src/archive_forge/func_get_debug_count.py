import html
import sys
import os
import traceback
from io import StringIO
import pprint
import itertools
import time
import re
from paste.exceptions import errormiddleware, formatter, collector
from paste import wsgilib
from paste import urlparser
from paste import httpexceptions
from paste import registry
from paste import request
from paste import response
from paste.evalexception import evalcontext
def get_debug_count(environ):
    """
    Return the unique debug count for the current request
    """
    if 'paste.evalexception.debug_count' in environ:
        return environ['paste.evalexception.debug_count']
    else:
        environ['paste.evalexception.debug_count'] = _next = next(debug_counter)
        return _next