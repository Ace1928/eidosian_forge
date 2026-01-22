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
def check_file_content(self, content, path):
    with open(path, 'r') as fd:
        ans = fd.read()
        self.assertEqual(content, ans.encode('latin-1'))