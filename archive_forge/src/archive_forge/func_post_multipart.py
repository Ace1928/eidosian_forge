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
def post_multipart(self, file):
    """Return a summary ("a * 65536
b * 65536") of the uploaded
                file.
                """
    contents = file.file.read()
    summary = []
    curchar = None
    count = 0
    for c in contents:
        if c == curchar:
            count += 1
        else:
            if count:
                curchar = chr(curchar)
                summary.append('%s * %d' % (curchar, count))
            count = 1
            curchar = c
    if count:
        curchar = chr(curchar)
        summary.append('%s * %d' % (curchar, count))
    return ', '.join(summary)