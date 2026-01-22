import io
import os
import sys
import re
import platform
import tempfile
import urllib.parse
import unittest.mock
from http.client import HTTPConnection
import pytest
import py.path
import path
import cherrypy
from cherrypy.lib import static
from cherrypy._cpcompat import HTTPSConnection, ntou, tonative
from cherrypy.test import helper
def error_page_404(status, message, traceback, version):
    path = os.path.join(curdir, 'static', '404.html')
    return static.serve_file(path, content_type='text/html')