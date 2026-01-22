import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_update(self):
    patch = {'op': 'replace', 'value': NEW_NODE_COUNT, 'path': '/node_count'}
    nodegroup = self.mgr.update(self.cluster_id, id=NODEGROUP1['id'], patch=patch)
    expect = [('PATCH', self.base_path + '%s' % NODEGROUP1['id'], {}, patch)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(NEW_NODE_COUNT, nodegroup.node_count)