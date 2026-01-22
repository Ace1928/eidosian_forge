import os
import platform
import threading
import time
from http.client import HTTPConnection
from distutils.spawn import find_executable
import pytest
from path import Path
from more_itertools import consume
import portend
import cherrypy
from cherrypy._cpcompat import HTTPSConnection
from cherrypy.lib import sessions
from cherrypy.lib import reprconf
from cherrypy.lib.httputil import response_codes
from cherrypy.test import helper
from cherrypy import _json as json
def test_5_Error_paths(self):
    self.getPage('/unknown/page')
    self.assertErrorPage(404, "The path '/unknown/page' was not found.")
    self.getPage('/restricted', self.cookies, method='POST')
    self.assertErrorPage(405, response_codes[405][1])