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
def test_changelog_handles_astrisk(self):
    self.repo.commit(message_content='Allow *.openstack.org to work')
    self.run_setup('sdist', allow_fail=False)
    with open(os.path.join(self.package_dir, 'ChangeLog'), 'r') as f:
        body = f.read()
    self.assertIn('\\*', body)