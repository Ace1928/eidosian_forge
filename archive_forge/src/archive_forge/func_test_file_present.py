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
def test_file_present(self):
    tmpfile = tempfile.mktemp()
    open(tmpfile, 'w')
    fileutils.delete_if_exists(tmpfile)
    self.assertFalse(os.path.exists(tmpfile))