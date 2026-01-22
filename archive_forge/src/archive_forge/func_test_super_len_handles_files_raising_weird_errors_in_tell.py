import os
import copy
import filecmp
from io import BytesIO
import tarfile
import zipfile
from collections import deque
import pytest
from requests import compat
from requests.cookies import RequestsCookieJar
from requests.structures import CaseInsensitiveDict
from requests.utils import (
from requests._internal_utils import unicode_is_ascii
from .compat import StringIO, cStringIO
@pytest.mark.parametrize('error', [IOError, OSError])
def test_super_len_handles_files_raising_weird_errors_in_tell(self, error):
    """If tell() raises errors, assume the cursor is at position zero."""

    class BoomFile(object):

        def __len__(self):
            return 5

        def tell(self):
            raise error()
    assert super_len(BoomFile()) == 0