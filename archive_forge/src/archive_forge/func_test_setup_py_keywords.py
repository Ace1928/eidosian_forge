import glob
import os
import sys
import tarfile
import fixtures
from pbr.tests import base
def test_setup_py_keywords(self):
    """setup.py --keywords.

        Test that the `./setup.py --keywords` command returns the correct
        value without balking.
        """
    self.run_setup('egg_info')
    stdout, _, _ = self.run_setup('--keywords')
    assert stdout == 'packaging, distutils, setuptools'