import os
import unittest
import unittest.mock as mock
from urllib.error import HTTPError
from distutils.command import upload as upload_mod
from distutils.command.upload import upload
from distutils.core import Distribution
from distutils.errors import DistutilsError
from distutils.log import ERROR, INFO
from distutils.tests.test_config import PYPIRC, BasePyPIRCCommandTestCase
def test_saved_password(self):
    self.write_file(self.rc, PYPIRC_NOPASSWORD)
    dist = Distribution()
    cmd = upload(dist)
    cmd.finalize_options()
    self.assertEqual(cmd.password, None)
    dist.password = 'xxx'
    cmd = upload(dist)
    cmd.finalize_options()
    self.assertEqual(cmd.password, 'xxx')