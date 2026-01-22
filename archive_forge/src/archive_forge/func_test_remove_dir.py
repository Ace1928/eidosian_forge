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
def test_remove_dir(self):
    tmpdir = tempfile.mktemp()
    os.mkdir(tmpdir)
    try:
        with fileutils.remove_path_on_error(tmpdir, lambda path: fileutils.delete_if_exists(path, os.rmdir)):
            raise Exception
    except Exception:
        self.assertFalse(os.path.exists(tmpdir))