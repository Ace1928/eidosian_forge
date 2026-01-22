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
def end_array(self, data):
    mark = self._marks.pop()
    self._stack[mark:] = [self._stack[mark:]]
    self._value = 0