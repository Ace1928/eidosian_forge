import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def testSlashes(self):
    self.getPage('/redirect?id=3')
    self.assertStatus(301)
    self.assertMatchesBody('<a href=([\'"])%s/redirect/[?]id=3\\1>%s/redirect/[?]id=3</a>' % (self.base(), self.base()))
    if self.prefix():
        self.getPage('')
        self.assertStatus(301)
        self.assertMatchesBody('<a href=([\'"])%s/\\1>%s/</a>' % (self.base(), self.base()))
    self.getPage('/redirect/by_code/?code=307')
    self.assertStatus(301)
    self.assertMatchesBody('<a href=([\'"])%s/redirect/by_code[?]code=307\\1>%s/redirect/by_code[?]code=307</a>' % (self.base(), self.base()))
    self.getPage('/url?path_info=page1')
    self.assertBody('%s/url/page1' % self.base())
    self.getPage('/url/leaf/?path_info=page1')
    self.assertBody('%s/url/page1' % self.base())