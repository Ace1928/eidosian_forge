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
def test_default_requirements(self):
    """Ensure default files used if no files provided."""
    tempdir = tempfile.mkdtemp()
    requirements = os.path.join(tempdir, 'requirements.txt')
    with open(requirements, 'w') as f:
        f.write('pbr')
    with mock.patch.object(packaging, 'REQUIREMENTS_FILES', (requirements,)):
        result = packaging.parse_requirements()
    self.assertEqual(['pbr'], result)