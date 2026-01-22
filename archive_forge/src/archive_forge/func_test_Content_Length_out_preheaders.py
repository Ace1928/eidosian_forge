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
def test_Content_Length_out_preheaders(self):
    self.persistent = True
    conn = self.HTTP_CONN
    conn.putrequest('GET', '/custom_cl?body=I+have+too+many+bytes&cl=5', skip_host=True)
    conn.putheader('Host', self.HOST)
    conn.endheaders()
    response = conn.getresponse()
    self.status, self.headers, self.body = webtest.shb(response)
    self.assertStatus(500)
    self.assertBody('The requested resource returned more bytes than the declared Content-Length.')
    conn.close()