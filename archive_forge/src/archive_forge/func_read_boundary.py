import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
def read_boundary(self):
    """Read the boundary headers defining a new range"""
    boundary_line = b'\r\n'
    while boundary_line == b'\r\n':
        boundary_line = self._file.readline()
    if boundary_line == b'':
        raise errors.HttpBoundaryMissing(self._path, self._boundary)
    if boundary_line != b'--' + self._boundary + b'\r\n':
        if self._unquote_boundary(boundary_line) != b'--' + self._boundary + b'\r\n':
            raise errors.InvalidHttpResponse(self._path, "Expected a boundary (%s) line, got '%s'" % (self._boundary, boundary_line))