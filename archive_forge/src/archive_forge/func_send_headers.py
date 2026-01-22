from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
def send_headers(self):
    """Transmit headers to the client, via self._write()"""
    self.cleanup_headers()
    self.headers_sent = True
    if not self.origin_server or self.client_is_modern():
        self.send_preamble()
        self._write(bytes(self.headers))