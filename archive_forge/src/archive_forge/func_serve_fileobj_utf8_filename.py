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
def serve_fileobj_utf8_filename(self):
    return static.serve_fileobj(io.BytesIO('☃\nfie\nfo\nfum'.encode('utf-8')), disposition='attachment', name='has_utf-8_character_☃.html')