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
def system_methodHelp(self, method_name):
    """system.methodHelp('add') => "Adds two integers together"

        Returns a string containing documentation for the specified method."""
    method = None
    if method_name in self.funcs:
        method = self.funcs[method_name]
    elif self.instance is not None:
        if hasattr(self.instance, '_methodHelp'):
            return self.instance._methodHelp(method_name)
        elif not hasattr(self.instance, '_dispatch'):
            try:
                method = resolve_dotted_attribute(self.instance, method_name, self.allow_dotted_names)
            except AttributeError:
                pass
    if method is None:
        return ''
    else:
        return pydoc.getdoc(method)