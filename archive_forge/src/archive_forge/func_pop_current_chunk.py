import contextlib
import re
import socket
import ssl
import zlib
from base64 import b64decode
from io import BufferedReader, BytesIO, TextIOWrapper
from test import onlyBrotlipy
import mock
import pytest
import six
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.response import HTTPResponse, brotli
from urllib3.util.response import is_fp_closed
from urllib3.util.retry import RequestHistory, Retry
def pop_current_chunk(self, amt=-1, till_crlf=False):
    if amt > 0 and till_crlf:
        raise ValueError("Can't specify amt and till_crlf.")
    if len(self.cur_chunk) <= 0:
        self.cur_chunk = self._pop_new_chunk()
    if till_crlf:
        try:
            i = self.cur_chunk.index(b'\r\n')
        except ValueError:
            self.cur_chunk = b''
            return b''
        else:
            chunk_part = self.cur_chunk[:i + 2]
            self.cur_chunk = self.cur_chunk[i + 2:]
            return chunk_part
    elif amt <= -1:
        chunk_part = self.cur_chunk
        self.cur_chunk = b''
        return chunk_part
    else:
        try:
            chunk_part = self.cur_chunk[:amt]
        except IndexError:
            chunk_part = self.cur_chunk
            self.cur_chunk = b''
        else:
            self.cur_chunk = self.cur_chunk[amt:]
        return chunk_part