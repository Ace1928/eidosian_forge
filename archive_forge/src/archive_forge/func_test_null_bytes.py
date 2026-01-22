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
@unittest.mock.patch('http.client._contains_disallowed_url_pchar_re', re.compile('[\\n]'), create=True)
def test_null_bytes(self):
    self.getPage('/static/\x00')
    self.assertStatus('404 Not Found')