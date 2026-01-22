import pytest
from dummyserver.testcase import (
from urllib3 import HTTPConnectionPool
from urllib3.util import SKIP_HEADER
from urllib3.util.retry import Retry
def test_preserve_transfer_encoding_header(self):
    self.start_chunked_handler()
    chunks = ['foo', 'bar', '', 'bazzzzzzzzzzzzzzzzzzzzzz']
    with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
        pool.urlopen('GET', '/', body=chunks, headers={'transfer-Encoding': 'test-transfer-encoding'}, chunked=True)
        te_headers = self._get_header_lines(b'transfer-encoding')
        assert len(te_headers) == 1
        assert te_headers[0] == b'transfer-encoding: test-transfer-encoding'