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
def test_install_writes_changelog(self):
    stdout, _, _ = self.run_setup('install', '--root', self.temp_dir + 'installed', allow_fail=False)
    self.expectThat(stdout, matchers.Contains('Generating ChangeLog'))