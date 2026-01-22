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
def make_eval_exception(app, global_conf, xmlhttp_key=None):
    """
    Wraps the application in an interactive debugger.

    This debugger is a major security hole, and should only be
    used during development.

    xmlhttp_key is a string that, if present in QUERY_STRING,
    indicates that the request is an XMLHttp request, and the
    Javascript/interactive debugger should not be returned.  (If you
    try to put the debugger somewhere with innerHTML, you will often
    crash the browser)
    """
    if xmlhttp_key is None:
        xmlhttp_key = global_conf.get('xmlhttp_key', '_')
    return EvalException(app, xmlhttp_key=xmlhttp_key)