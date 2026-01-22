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
def testHookErrors(self):
    self.getPage('/demo/?id=1')
    self.assertBody('A horrorshow lomtick of cherry 3.14159')
    time.sleep(0.1)
    self.getPage('/demo/ended/1')
    self.assertBody('True')
    valerr = '\n    raise ValueError()\nValueError'
    self.getPage('/demo/err?id=3')
    self.assertErrorPage(502, pattern=valerr)
    time.sleep(0.1)
    self.getPage('/demo/ended/3')
    self.assertBody('True')
    if cherrypy.server.protocol_version == 'HTTP/1.0' or getattr(cherrypy.server, 'using_apache', False):
        self.getPage('/demo/errinstream?id=5')
        self.assertStatus('200 OK')
        self.assertBody('nonconfidential')
    else:
        self.assertRaises((ValueError, IncompleteRead), self.getPage, '/demo/errinstream?id=5')
    time.sleep(0.1)
    self.getPage('/demo/ended/5')
    self.assertBody('True')
    self.getPage('/demo/restricted')
    self.assertErrorPage(401)
    self.getPage('/demo/userid')
    self.assertBody('Welcome!')