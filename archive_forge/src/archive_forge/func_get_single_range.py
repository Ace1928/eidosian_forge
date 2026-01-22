import errno
import http.client as http_client
import http.server as http_server
import os
import posixpath
import random
import re
import socket
import sys
from urllib.parse import urlparse
from .. import osutils, urlutils
from . import test_server
def get_single_range(self, file, file_size, start, end):
    self.send_response(206)
    length = end - start + 1
    self.send_header('Accept-Ranges', 'bytes')
    self.send_header('Content-Length', '%d' % length)
    self.send_header('Content-Type', 'application/octet-stream')
    self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, file_size))
    self.end_headers()
    self.send_range_content(file, start, length)