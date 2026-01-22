import itertools
import platform
import threading
from http.client import HTTPConnection
import cherrypy
from cherrypy._cpcompat import HTTPSConnection
from cherrypy.test import helper
def getpage():
    host = '%s:%s' % (self.interface(), self.PORT)
    if self.scheme == 'https':
        c = HTTPSConnection(host)
    else:
        c = HTTPConnection(host)
    try:
        c.putrequest('GET', '/')
        c.endheaders()
        response = c.getresponse()
        body = response.read()
        self.assertEqual(response.status, 200)
        self.assertEqual(body, b'Hello world!')
    finally:
        c.close()
    next(success)