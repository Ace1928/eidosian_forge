import logging
import mock
import os
import re
import sys
import shutil
import tempfile
import textwrap
import unittest
from gae_ext_runtime import testutil
from gae_ext_runtime import ext_runtime
import constants
def test_java_files_no_java(self):
    self.write_file('foo.nojava', '')
    self.assertFalse(self.generate_configs())
    self.assertEqual(os.listdir(self.temp_path), ['foo.nojava'])