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
def testGuaranteedHooks(self):
    self.getPage('/demo/err_in_onstart')
    self.assertErrorPage(502)
    tmpl = "AttributeError: 'str' object has no attribute '{attr}'"
    expected_msg = tmpl.format(attr='items')
    self.assertInBody(expected_msg)