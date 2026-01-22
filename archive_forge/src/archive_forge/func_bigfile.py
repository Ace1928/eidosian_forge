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
@cherrypy.config(**{'response.stream': True})
def bigfile(self):
    self.f = static.serve_file(bigfile_filepath)
    return self.f