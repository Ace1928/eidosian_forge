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
def dump_bool(self, value, write):
    write('<value><boolean>')
    write(value and '1' or '0')
    write('</boolean></value>\n')