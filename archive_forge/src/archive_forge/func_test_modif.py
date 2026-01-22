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
def test_modif(self):
    self.getPage('/static/dirback.jpg')
    self.assertStatus('200 OK')
    lastmod = ''
    for k, v in self.headers:
        if k == 'Last-Modified':
            lastmod = v
    ims = ('If-Modified-Since', lastmod)
    self.getPage('/static/dirback.jpg', headers=[ims])
    self.assertStatus(304)
    self.assertNoHeader('Content-Type')
    self.assertNoHeader('Content-Length')
    self.assertNoHeader('Content-Disposition')
    self.assertBody('')