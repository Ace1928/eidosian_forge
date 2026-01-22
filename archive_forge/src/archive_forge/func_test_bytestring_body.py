import pytest
from dummyserver.testcase import (
from urllib3 import HTTPConnectionPool
from urllib3.util import SKIP_HEADER
from urllib3.util.retry import Retry
def test_bytestring_body(self):
    self._test_body(b'thisshouldbeonechunk\r\nasdf')