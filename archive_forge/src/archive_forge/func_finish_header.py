from a file, a socket or a WSGI environment. The parser can be used to replace
import re
import sys
from io import BytesIO
from tempfile import TemporaryFile
from urllib.parse import parse_qs
from wsgiref.headers import Headers
from collections.abc import MutableMapping as DictMixin
def finish_header(self):
    self.file = BytesIO()
    self.headers = Headers(self.headerlist)
    content_disposition = self.headers.get('Content-Disposition', '')
    content_type = self.headers.get('Content-Type', '')
    if not content_disposition:
        raise MultipartError('Content-Disposition header is missing.')
    self.disposition, self.options = parse_options_header(content_disposition)
    self.name = self.options.get('name')
    self.filename = self.options.get('filename')
    self.content_type, options = parse_options_header(content_type)
    self.charset = options.get('charset') or self.charset
    self.content_length = int(self.headers.get('Content-Length', '-1'))