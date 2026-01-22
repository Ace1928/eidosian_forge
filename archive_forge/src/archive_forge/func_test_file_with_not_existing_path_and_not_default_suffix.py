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
def test_file_with_not_existing_path_and_not_default_suffix(self):
    suffix = '.txt'
    random_dir = uuid.uuid4().hex
    path = '/tmp/%s/test2' % random_dir
    res = fileutils.write_to_tempfile(self.content, path=path, suffix=suffix)
    self.assertTrue(os.path.exists(res))
    basepath, tmpfile = os.path.split(res)
    self.assertTrue(tmpfile.startswith('tmp'))
    self.assertEqual(basepath, path)
    self.assertTrue(tmpfile.endswith(suffix))
    self.check_file_content(res)
    shutil.rmtree('/tmp/' + random_dir)