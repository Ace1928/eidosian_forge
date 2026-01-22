import os
import sys
import time
import shutil
import platform
import tempfile
import unittest
import multiprocessing
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
def make_tmp_file(self, content=None):
    if not content:
        content = b'blah' * 1024
    _, tmppath = tempfile.mkstemp()
    with open(tmppath, 'wb') as fp:
        fp.write(content)
    return tmppath