import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_create_fail(self):
    CREATE_CLUSTER_FAIL = copy.deepcopy(CREATE_CLUSTER)
    CREATE_CLUSTER_FAIL['wrong_key'] = 'wrong'
    self.assertRaisesRegex(exceptions.InvalidAttribute, 'Key must be in %s' % ','.join(clusters.CREATION_ATTRIBUTES), self.mgr.create, **CREATE_CLUSTER_FAIL)
    self.assertEqual([], self.api.calls)