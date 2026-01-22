import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_list_with_limit(self):
    expect = [('GET', self.base_path + '?limit=2', {}, None)]
    self._test_nodegroup_list_with_filters(self.cluster_id, limit=2, expect=expect)