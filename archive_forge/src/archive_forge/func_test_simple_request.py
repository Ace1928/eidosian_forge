import os
import socket
import atexit
import tempfile
from http.client import HTTPConnection
import pytest
import cherrypy
from cherrypy.test import helper
def test_simple_request(self):
    self.getPage('/')
    self.assertStatus('200 OK')
    self.assertInBody('Test OK')