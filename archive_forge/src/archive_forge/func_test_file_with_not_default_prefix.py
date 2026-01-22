import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from unittest import mock
import uuid
import yaml
from oslotest import base as test_base
from oslo_utils import fileutils
def test_file_with_not_default_prefix(self):
    prefix = 'test'
    res = fileutils.write_to_tempfile(self.content, prefix=prefix)
    self.assertTrue(os.path.exists(res))
    basepath, tmpfile = os.path.split(res)
    self.assertTrue(tmpfile.startswith(prefix))
    self.assertTrue(basepath.startswith(tempfile.gettempdir()))
    self.check_file_content(res)