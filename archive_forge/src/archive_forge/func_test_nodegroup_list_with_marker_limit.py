import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_list_with_marker_limit(self):
    filter_ = '?limit=2&marker=%s' % NODEGROUP2['uuid']
    expect = [('GET', self.base_path + filter_, {}, None)]
    self._test_nodegroup_list_with_filters(self.cluster_id, limit=2, marker=NODEGROUP2['uuid'], expect=expect)