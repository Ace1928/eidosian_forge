from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
def test_header_presence(self):
    self.getPage('/headers/Content-Type', headers=[])
    self.assertStatus(500)
    self.getPage('/headers/Content-Type', headers=[('Content-type', 'application/json')])
    self.assertBody('application/json')