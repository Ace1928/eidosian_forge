import json
import os.path
import shutil
import tempfile
import unittest
import mock
import six
from apitools.base.py import credentials_lib
from apitools.base.py import util
def testGetAdcNone(self):
    creds = credentials_lib._GetApplicationDefaultCredentials(client_info={'scope': ''})
    self.assertIsNone(creds)