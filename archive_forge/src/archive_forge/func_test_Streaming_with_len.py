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
def test_Streaming_with_len(self):
    try:
        self._streaming(set_cl=True)
    finally:
        try:
            self.HTTP_CONN.close()
        except (TypeError, AttributeError):
            pass