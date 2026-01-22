import gzip
import io
import sys
import time
import types
import unittest
import operator
from http.client import IncompleteRead
import cherrypy
from cherrypy import tools
from cherrypy._cpcompat import ntou
from cherrypy.test import helper, _test_decorators
@cherrypy.expose
@cherrypy.config(**{'tools.streamer.on': True, 'tools.streamer.arg': 'arg value'})
def tarfile(self):
    actual = cherrypy.request.config.get('tools.streamer.arg')
    assert actual == 'arg value'
    cherrypy.response.output.write(b'I am ')
    cherrypy.response.output.write(b'a tarfile')