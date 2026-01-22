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
def dump_unicode(self, value, write, escape=escape):
    write('<value><string>')
    write(escape(value))
    write('</string></value>\n')