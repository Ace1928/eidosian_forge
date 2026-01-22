import copy
import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import quotas
def test_show_project_resource_quota(self):
    expect = [('GET', '/v1/quotas/%(id)s/%(res)s' % {'id': QUOTA2['project_id'], 'res': QUOTA2['resource']}, {}, None)]
    quotas = self.mgr.get(QUOTA2['project_id'], QUOTA2['resource'])
    self.assertEqual(expect, self.api.calls)
    expected_quotas = QUOTA2
    self.assertEqual(expected_quotas, quotas._info)