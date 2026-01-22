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
def log_message(self, format, *args):
    tcs = self.server.test_case_server
    tcs.log('webserver - %s - - [%s] %s "%s" "%s"', self.address_string(), self.log_date_time_string(), format % args, self.headers.get('referer', '-'), self.headers.get('user-agent', '-'))