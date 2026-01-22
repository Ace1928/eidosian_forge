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
def testDecorator(self):

    @cherrypy.tools.register('on_start_resource')
    def example():
        pass
    self.assertTrue(isinstance(cherrypy.tools.example, cherrypy.Tool))
    self.assertEqual(cherrypy.tools.example._point, 'on_start_resource')

    @cherrypy.tools.register('before_finalize', name='renamed', priority=60)
    def example():
        pass
    self.assertTrue(isinstance(cherrypy.tools.renamed, cherrypy.Tool))
    self.assertEqual(cherrypy.tools.renamed._point, 'before_finalize')
    self.assertEqual(cherrypy.tools.renamed._name, 'renamed')
    self.assertEqual(cherrypy.tools.renamed._priority, 60)