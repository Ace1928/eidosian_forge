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
def post_filename(self, myfile):
    """Return the name of the file which was uploaded."""
    return myfile.filename