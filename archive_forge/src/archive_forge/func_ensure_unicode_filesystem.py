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
def ensure_unicode_filesystem():
    """
    TODO: replace with simply pytest fixtures once webtest.TestCase
    no longer implies unittest.
    """
    tmpdir = py.path.local(tempfile.mkdtemp())
    try:
        _check_unicode_filesystem(tmpdir)
    finally:
        tmpdir.remove()