import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_list_with_limit(self):
    expect = [('GET', '/v1/clusters/?limit=2', {}, None)]
    self._test_cluster_list_with_filters(limit=2, expect=expect)