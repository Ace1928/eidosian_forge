import os
import socket
import atexit
import tempfile
from http.client import HTTPConnection
import pytest
import cherrypy
from cherrypy.test import helper
def usocket_path():
    fd, path = tempfile.mkstemp('cp_test.sock')
    os.close(fd)
    os.remove(path)
    return path