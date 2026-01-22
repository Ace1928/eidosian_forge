import gzip
import io
from paste.response import header_value, remove_header
from paste.httpheaders import CONTENT_LENGTH
def gzip_start_response(self, status, headers, exc_info=None):
    self.headers = headers
    ct = header_value(headers, 'content-type')
    ce = header_value(headers, 'content-encoding')
    self.compressible = False
    if ct and (ct.startswith('text/') or ct.startswith('application/')) and ('zip' not in ct):
        self.compressible = True
    if ce:
        self.compressible = False
    if self.compressible:
        headers.append(('content-encoding', 'gzip'))
    remove_header(headers, 'content-length')
    self.headers = headers
    self.status = status
    return self.buffer.write