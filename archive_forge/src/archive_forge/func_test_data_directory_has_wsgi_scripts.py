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
def test_data_directory_has_wsgi_scripts(self):
    scripts_dir = os.path.join(self.extracted_wheel_dir, 'pbr_testpackage-0.0.data/scripts')
    self.assertTrue(os.path.exists(scripts_dir))
    scripts = os.listdir(scripts_dir)
    self.assertIn('pbr_test_wsgi', scripts)
    self.assertIn('pbr_test_wsgi_with_class', scripts)
    self.assertNotIn('pbr_test_cmd', scripts)
    self.assertNotIn('pbr_test_cmd_with_class', scripts)