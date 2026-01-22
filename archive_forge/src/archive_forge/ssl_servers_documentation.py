from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import filter, str
from future import utils
import os
import sys
import ssl
import pprint
import socket
from future.backports.urllib import parse as urllib_parse
from future.backports.http.server import (HTTPServer as _HTTPServer,
from future.backports.test import support
Serve a HEAD request.