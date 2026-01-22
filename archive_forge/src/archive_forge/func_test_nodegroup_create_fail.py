import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_create_fail(self):
    CREATE_NODEGROUP_FAIL = copy.deepcopy(CREATE_NODEGROUP)
    CREATE_NODEGROUP_FAIL['wrong_key'] = 'wrong'
    self.assertRaisesRegex(exceptions.InvalidAttribute, 'Key must be in %s' % ','.join(nodegroups.CREATION_ATTRIBUTES), self.mgr.create, self.cluster_id, **CREATE_NODEGROUP_FAIL)
    self.assertEqual([], self.api.calls)