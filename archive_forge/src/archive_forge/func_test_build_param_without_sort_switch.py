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
def test_build_param_without_sort_switch(self):
    dict_param = {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
    result = utils.build_query_param(dict_param, True)
    self.assertIn('key1=val1', result)
    self.assertIn('key2=val2', result)
    self.assertIn('key3=val3', result)