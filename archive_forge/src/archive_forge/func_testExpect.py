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
def testExpect(self):
    e = ('Expect', '100-continue')
    self.getPage('/headerelements/get_elements?headername=Expect', [e])
    self.assertBody('100-continue')
    self.getPage('/expect/expectation_failed', [e])
    self.assertStatus(417)