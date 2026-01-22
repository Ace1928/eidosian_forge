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
def test_untagged_version_after_pre_has_dev_version_preversion(self):
    self.repo.commit()
    self.repo.tag('1.2.3.0a1')
    self.repo.commit()
    version = packaging._get_version_from_git('1.2.5')
    self.assertThat(version, matchers.StartsWith('1.2.5.dev1'))