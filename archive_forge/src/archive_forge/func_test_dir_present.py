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
def test_dir_present(self):
    tmpdir = tempfile.mktemp()
    os.mkdir(tmpdir)
    fileutils.delete_if_exists(tmpdir, remove=os.rmdir)
    self.assertFalse(os.path.exists(tmpdir))