import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_list(self):
    clusters = self.mgr.list(self.cluster_id)
    expect = [('GET', self.base_path, {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(clusters, matchers.HasLength(2))