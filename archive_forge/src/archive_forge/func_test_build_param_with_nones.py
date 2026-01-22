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
@ddt.data({}, None, {'key1': 'val1', 'key2': None, 'key3': False, 'key4': ''})
def test_build_param_with_nones(self, dict_param):
    result = utils.build_query_param(dict_param)
    expected = ('key1=val1', 'key3=False') if dict_param else ()
    for exp in expected:
        self.assertIn(exp, result)
    if not expected:
        self.assertEqual('', result)