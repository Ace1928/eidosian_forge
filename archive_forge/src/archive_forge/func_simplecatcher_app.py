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
def simplecatcher_app(environ, start_response):
    try:
        return application(environ, start_response)
    except:
        out = StringIO()
        traceback.print_exc(file=out)
        start_response('500 Server Error', [('content-type', 'text/html')], sys.exc_info())
        res = out.getvalue()
        return ['<h3>Error</h3><pre>%s</pre>' % html_quote(res)]