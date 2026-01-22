from xmlrpc.client import Fault, dumps, loads, gzip_encode, gzip_decode
from http.server import BaseHTTPRequestHandler
from functools import partial
from inspect import signature
import html
import http.server
import socketserver
import sys
import os
import re
import pydoc
import traceback
def docserver(self, server_name, package_documentation, methods):
    """Produce HTML documentation for an XML-RPC server."""
    fdict = {}
    for key, value in methods.items():
        fdict[key] = '#-' + key
        fdict[value] = fdict[key]
    server_name = self.escape(server_name)
    head = '<big><big><strong>%s</strong></big></big>' % server_name
    result = self.heading(head)
    doc = self.markup(package_documentation, self.preformat, fdict)
    doc = doc and '<tt>%s</tt>' % doc
    result = result + '<p>%s</p>\n' % doc
    contents = []
    method_items = sorted(methods.items())
    for key, value in method_items:
        contents.append(self.docroutine(value, key, funcs=fdict))
    result = result + self.bigsection('Methods', 'functions', ''.join(contents))
    return result