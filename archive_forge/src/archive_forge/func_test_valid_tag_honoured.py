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
def test_valid_tag_honoured(self):
    self.repo.commit()
    self.repo.tag('1.3.0.0a1')
    version = packaging._get_version_from_git()
    self.assertEqual('1.3.0.0a1', version)