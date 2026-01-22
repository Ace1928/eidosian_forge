import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def bare_put_file_non_atomic():
    response = self.request('PUT', abspath, body=bytes, headers=headers)
    code = response.status
    if code in (403, 404, 409):
        raise transport.NoSuchFile(abspath)
    elif code not in (200, 201, 204):
        raise self._raise_http_error(abspath, response, 'put file failed')