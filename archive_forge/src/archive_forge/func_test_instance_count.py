import os
import testtools
from unittest import mock
from troveclient.v1 import modules
def test_instance_count(self):
    expected_query = {'include_clustered': True, 'count_only': True}
    self._test_instances(expected_query)