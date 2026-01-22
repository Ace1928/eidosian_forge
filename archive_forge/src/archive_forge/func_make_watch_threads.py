import sys
import cgi
import time
import traceback
from io import StringIO
from thread import get_ident
from paste import httpexceptions
from paste.request import construct_url, parse_formvars
from paste.util.template import HTMLTemplate, bunch
def make_watch_threads(global_conf, allow_kill=False):
    from paste.deploy.converters import asbool
    return WatchThreads(allow_kill=asbool(allow_kill))