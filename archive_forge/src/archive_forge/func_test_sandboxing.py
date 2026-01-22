import os
import pytest
from tempfile import mkdtemp, mkstemp, NamedTemporaryFile
from shutil import rmtree
import numpy.lib._datasource as datasource
from numpy.testing import assert_, assert_equal, assert_raises
import urllib.request as urllib_request
from urllib.parse import urlparse
from urllib.error import URLError
def test_sandboxing(self):
    tmp_path = lambda x: os.path.abspath(self.repos.abspath(x))
    assert_(tmp_path(valid_httpfile()).startswith(self.tmpdir))
    for fn in malicious_files:
        assert_(tmp_path(http_path + fn).startswith(self.tmpdir))
        assert_(tmp_path(fn).startswith(self.tmpdir))