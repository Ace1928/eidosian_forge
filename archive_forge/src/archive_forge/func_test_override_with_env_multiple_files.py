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
def test_override_with_env_multiple_files(self):
    _, tmp_file = tempfile.mkstemp(prefix='openstack', suffix='.setup')
    with open(tmp_file, 'w') as fh:
        fh.write('foo\nbar')
    self.useFixture(fixtures.EnvironmentVariable('PBR_REQUIREMENTS_FILES', 'no-such-file,' + tmp_file))
    self.assertEqual(['foo', 'bar'], packaging.parse_requirements())