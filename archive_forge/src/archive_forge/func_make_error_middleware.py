import sys
import traceback
import cgi
from io import StringIO
from paste.exceptions import formatter, collector, reporter
from paste import wsgilib
from paste import request
def make_error_middleware(app, global_conf, **kw):
    return ErrorMiddleware(app, global_conf=global_conf, **kw)