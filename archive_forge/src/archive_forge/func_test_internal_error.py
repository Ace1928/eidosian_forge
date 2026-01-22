import os
import socket
import atexit
import tempfile
from http.client import HTTPConnection
import pytest
import cherrypy
from cherrypy.test import helper
def test_internal_error(self):
    self.getPage('/error')
    self.assertStatus('500 Internal Server Error')
    self.assertInBody('Invalid page')