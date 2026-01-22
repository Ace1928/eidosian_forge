import datetime
from itertools import count
import os
import threading
import time
import urllib.parse
import pytest
import cherrypy
from cherrypy.lib import httputil
from cherrypy.test import helper
def testLastModified(self):
    self.getPage('/a.gif')
    self.assertStatus(200)
    self.assertBody(gif_bytes)
    lm1 = self.assertHeader('Last-Modified')
    self.getPage('/a.gif')
    self.assertStatus(200)
    self.assertBody(gif_bytes)
    self.assertHeader('Age')
    lm2 = self.assertHeader('Last-Modified')
    self.assertEqual(lm1, lm2)
    self.getPage('/a.gif', [('If-Modified-Since', lm1)])
    self.assertStatus(304)
    self.assertNoHeader('Last-Modified')
    if not getattr(cherrypy.server, 'using_apache', False):
        self.assertHeader('Age')