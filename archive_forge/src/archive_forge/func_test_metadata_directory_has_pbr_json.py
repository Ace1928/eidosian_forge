import email
import email.errors
import os
import re
import sysconfig
import tempfile
import textwrap
import fixtures
import pkg_resources
import six
import testscenarios
import testtools
from testtools import matchers
import virtualenv
from wheel import wheelfile
from pbr import git
from pbr import packaging
from pbr.tests import base
def test_metadata_directory_has_pbr_json(self):
    pbr_json = os.path.join(self.extracted_wheel_dir, 'pbr_testpackage-0.0.dist-info/pbr.json')
    self.assertTrue(os.path.exists(pbr_json))