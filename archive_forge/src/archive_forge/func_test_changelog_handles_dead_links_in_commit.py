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
def test_changelog_handles_dead_links_in_commit(self):
    self.repo.commit(message_content='See os_ for to_do about qemu_.')
    self.run_setup('sdist', allow_fail=False)
    with open(os.path.join(self.package_dir, 'ChangeLog'), 'r') as f:
        body = f.read()
    self.assertIn('os\\_', body)
    self.assertIn('to\\_do', body)
    self.assertIn('qemu\\_', body)