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
def testEndRequestOnDrop(self):
    old_timeout = None
    try:
        httpserver = cherrypy.server.httpserver
        old_timeout = httpserver.timeout
    except (AttributeError, IndexError):
        return self.skip()
    try:
        httpserver.timeout = timeout
        self.persistent = True
        try:
            conn = self.HTTP_CONN
            conn.putrequest('GET', '/demo/stream?id=9', skip_host=True)
            conn.putheader('Host', self.HOST)
            conn.endheaders()
        finally:
            self.persistent = False
        time.sleep(timeout * 2)
        self.getPage('/demo/ended/9')
        self.assertBody('True')
    finally:
        if old_timeout is not None:
            httpserver.timeout = old_timeout