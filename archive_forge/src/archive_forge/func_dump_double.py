import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
def dump_double(self, value, write):
    write('<value><double>')
    write(repr(value))
    write('</double></value>\n')