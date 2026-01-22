import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
def test_empty_line_formatting(self):
    description = 'Check Summary \n\nThis is a description to check that empty line has no white spaces.'
    expected = '# Check Summary\n#\n# This is a description to check that empty line has no white spaces.\n#"admin": "is_admin:True"\n\n'
    self._test_formatting(description, expected)