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
def dump_nil(self, value, write):
    if not self.allow_none:
        raise TypeError('cannot marshal None unless allow_none is enabled')
    write('<value><nil/></value>')