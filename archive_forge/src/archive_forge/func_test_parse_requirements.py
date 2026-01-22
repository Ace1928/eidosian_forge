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
def test_parse_requirements(self):
    tmp_file = tempfile.NamedTemporaryFile()
    req_string = self.url
    if hasattr(self, 'editable') and self.editable:
        req_string = '-e %s' % req_string
    if hasattr(self, 'versioned') and self.versioned:
        req_string = '%s-1.2.3' % req_string
    if hasattr(self, 'has_subdirectory') and self.has_subdirectory:
        req_string = '%s&subdirectory=baz' % req_string
    with open(tmp_file.name, 'w') as fh:
        fh.write(req_string)
    self.assertEqual(self.expected, packaging.parse_requirements([tmp_file.name]))