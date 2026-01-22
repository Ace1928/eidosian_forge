import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
def set_range_from_header(self, content_range):
    """Helper to set the new range from its description in the headers"""
    try:
        rtype, values = content_range.split()
    except ValueError:
        raise errors.InvalidHttpRange(self._path, content_range, 'Malformed header')
    if rtype != 'bytes':
        raise errors.InvalidHttpRange(self._path, content_range, "Unsupported range type '%s'" % rtype)
    try:
        start_end, total = values.split('/')
        start, end = start_end.split('-')
        start = int(start)
        end = int(end)
    except ValueError:
        raise errors.InvalidHttpRange(self._path, content_range, 'Invalid range values')
    size = end - start + 1
    if size <= 0:
        raise errors.InvalidHttpRange(self._path, content_range, 'Invalid range, size <= 0')
    self.set_range(start, size)