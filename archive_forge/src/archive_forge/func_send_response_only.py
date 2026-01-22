import copy
import datetime
import email.utils
import html
import http.client
import io
import itertools
import mimetypes
import os
import posixpath
import select
import shutil
import socket # For gethostbyaddr()
import socketserver
import sys
import time
import urllib.parse
from http import HTTPStatus
def send_response_only(self, code, message=None):
    """Send the response header only."""
    if self.request_version != 'HTTP/0.9':
        if message is None:
            if code in self.responses:
                message = self.responses[code][0]
            else:
                message = ''
        if not hasattr(self, '_headers_buffer'):
            self._headers_buffer = []
        self._headers_buffer.append(('%s %d %s\r\n' % (self.protocol_version, code, message)).encode('latin-1', 'strict'))