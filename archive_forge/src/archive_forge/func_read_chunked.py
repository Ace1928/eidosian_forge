from __future__ import absolute_import
import io
import logging
import zlib
from contextlib import contextmanager
from socket import error as SocketError
from socket import timeout as SocketTimeout
from ._collections import HTTPHeaderDict
from .connection import BaseSSLError, HTTPException
from .exceptions import (
from .packages import six
from .util.response import is_fp_closed, is_response_to_head
def read_chunked(self, amt=None, decode_content=None):
    """
        Similar to :meth:`HTTPResponse.read`, but with an additional
        parameter: ``decode_content``.

        :param amt:
            How much of the content to read. If specified, caching is skipped
            because it doesn't make sense to cache partial content as the full
            response.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.
        """
    self._init_decoder()
    if not self.chunked:
        raise ResponseNotChunked("Response is not chunked. Header 'transfer-encoding: chunked' is missing.")
    if not self.supports_chunked_reads():
        raise BodyNotHttplibCompatible('Body should be http.client.HTTPResponse like. It should have have an fp attribute which returns raw chunks.')
    with self._error_catcher():
        if self._original_response and is_response_to_head(self._original_response):
            self._original_response.close()
            return
        if self._fp.fp is None:
            return
        while True:
            self._update_chunk_length()
            if self.chunk_left == 0:
                break
            chunk = self._handle_chunk(amt)
            decoded = self._decode(chunk, decode_content=decode_content, flush_decoder=False)
            if decoded:
                yield decoded
        if decode_content:
            decoded = self._flush_decoder()
            if decoded:
                yield decoded
        while True:
            line = self._fp.fp.readline()
            if not line:
                break
            if line == b'\r\n':
                break
        if self._original_response:
            self._original_response.close()