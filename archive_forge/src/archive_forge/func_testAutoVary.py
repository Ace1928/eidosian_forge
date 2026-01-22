import os
import cherrypy
from cherrypy import tools
from cherrypy.test import helper
def testAutoVary(self):
    self.getPage('/autovary/')
    self.assertHeader('Vary', 'Accept, Accept-Charset, Accept-Encoding, Host, If-Modified-Since, Range')