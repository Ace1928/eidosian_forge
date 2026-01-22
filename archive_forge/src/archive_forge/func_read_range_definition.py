import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
def read_range_definition(self):
    """Read a new range definition in a multi parts message.

        Parse the headers including the empty line following them so that we
        are ready to read the data itself.
        """
    self._headers = http_client.parse_headers(self._file)
    content_range = self._headers.get('content-range', None)
    if content_range is None:
        raise errors.InvalidHttpResponse(self._path, 'Content-Range header missing in a multi-part response', headers=self._headers)
    self.set_range_from_header(content_range)