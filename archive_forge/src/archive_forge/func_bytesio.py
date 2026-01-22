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
@cherrypy.expose
def bytesio(self):
    f = io.BytesIO(b'Fee\nfie\nfo\nfum')
    return static.serve_fileobj(f, content_type='text/plain')