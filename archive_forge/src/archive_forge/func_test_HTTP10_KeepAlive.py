import errno
import socket
import sys
import time
import urllib.parse
from http.client import BadStatusLine, HTTPConnection, NotConnected
from cheroot.test import webtest
import cherrypy
from cherrypy._cpcompat import HTTPSConnection, ntob, tonative
from cherrypy.test import helper
def test_HTTP10_KeepAlive(self):
    self.PROTOCOL = 'HTTP/1.0'
    if self.scheme == 'https':
        self.HTTP_CONN = HTTPSConnection
    else:
        self.HTTP_CONN = HTTPConnection
    self.getPage('/page2')
    self.assertStatus('200 OK')
    self.assertBody(pov)
    self.persistent = True
    self.getPage('/page3', headers=[('Connection', 'Keep-Alive')])
    self.assertStatus('200 OK')
    self.assertBody(pov)
    self.assertHeader('Connection', 'Keep-Alive')
    self.getPage('/page3')
    self.assertStatus('200 OK')
    self.assertBody(pov)