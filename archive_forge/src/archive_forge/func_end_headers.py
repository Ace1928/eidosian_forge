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
def end_headers(self):
    """Send the blank line ending the MIME headers."""
    if self.request_version != 'HTTP/0.9':
        self._headers_buffer.append(b'\r\n')
        self.flush_headers()