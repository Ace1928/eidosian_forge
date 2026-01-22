import errno
import mimetypes
import socket
import sys
from unittest import mock
import urllib.parse
from http.client import HTTPConnection
import cherrypy
from cherrypy._cpcompat import HTTPSConnection
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'request.process_request_body': False})
def no_body(self, *args, **kwargs):
    return 'Hello world!'