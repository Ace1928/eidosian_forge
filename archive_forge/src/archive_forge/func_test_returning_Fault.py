import sys
import socket
from xmlrpc.client import (
import cherrypy
from cherrypy import _cptools
from cherrypy.test import helper
@cherrypy.expose
def test_returning_Fault(self):
    return Fault(1, 'custom Fault response')