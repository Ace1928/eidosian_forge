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
def flush_headers(self):
    if hasattr(self, '_headers_buffer'):
        self.wfile.write(b''.join(self._headers_buffer))
        self._headers_buffer = []