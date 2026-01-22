from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
def test_CONNECT_method(self):
    self.persistent = True
    try:
        conn = self.HTTP_CONN
        conn.request('CONNECT', 'created.example.com:3128')
        response = conn.response_class(conn.sock, method='CONNECT')
        response.begin()
        self.assertEqual(response.status, 204)
    finally:
        self.persistent = False
    self.persistent = True
    try:
        conn = self.HTTP_CONN
        conn.request('CONNECT', 'body.example.com:3128')
        response = conn.response_class(conn.sock, method='CONNECT')
        response.begin()
        self.assertEqual(response.status, 200)
        self.body = response.read()
        self.assertBody(b'CONNECTed to /body.example.com:3128')
    finally:
        self.persistent = False