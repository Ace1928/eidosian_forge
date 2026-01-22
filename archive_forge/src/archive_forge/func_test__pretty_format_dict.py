import collections
import io
import sys
from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient.tests.unit import utils as test_utils
from cinderclient import utils
def test__pretty_format_dict(self):
    content = {'key1': 'value1', 'key2': 'value2'}
    expected = 'key1 : value1\nkey2 : value2'
    result = shell_utils._pretty_format_dict(content)
    self.assertEqual(expected, result)