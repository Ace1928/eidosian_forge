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
def test_is_json(self):
    self.assertTrue(fileutils.is_json(self.json_file))
    self.assertFalse(fileutils.is_json(self.yaml_file))