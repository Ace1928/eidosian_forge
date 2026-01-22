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
def test_tarfile_member(self, tmpdir):
    file_obj = tmpdir.join('test.txt')
    file_obj.write('Test')
    tar_obj = str(tmpdir.join('test.tar'))
    with tarfile.open(tar_obj, 'w') as tar:
        tar.add(str(file_obj), arcname='test.txt')
    with tarfile.open(tar_obj) as tar:
        member = tar.extractfile('test.txt')
        assert super_len(member) == 4