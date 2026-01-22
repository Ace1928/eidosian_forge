import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_delete_by_name(self):
    nodegroup = self.mgr.delete(self.cluster_id, NODEGROUP1['name'])
    expect = [('DELETE', self.base_path + '%s' % NODEGROUP1['name'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(nodegroup)