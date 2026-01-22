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
def test_error_page_with_serve_file(self):
    self.getPage('/404test/yunyeen')
    self.assertStatus(404)
    self.assertInBody("I couldn't find that thing")