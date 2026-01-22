import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def html_topicpage(topic):
    """Topic or keyword help page."""
    buf = io.StringIO()
    htmlhelp = Helper(buf, buf)
    contents, xrefs = htmlhelp._gettopic(topic)
    if topic in htmlhelp.keywords:
        title = 'KEYWORD'
    else:
        title = 'TOPIC'
    heading = html.heading('<strong class="title">%s</strong>' % title)
    contents = '<pre>%s</pre>' % html.markup(contents)
    contents = html.bigsection(topic, 'index', contents)
    if xrefs:
        xrefs = sorted(xrefs.split())

        def bltinlink(name):
            return '<a href="topic?key=%s">%s</a>' % (name, name)
        xrefs = html.multicolumn(xrefs, bltinlink)
        xrefs = html.section('Related help topics: ', 'index', xrefs)
    return ('%s %s' % (title, topic), ''.join((heading, contents, xrefs)))