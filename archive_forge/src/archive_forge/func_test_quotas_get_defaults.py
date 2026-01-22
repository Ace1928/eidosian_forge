import testtools
from zunclient.tests.unit import utils
from zunclient.v1 import quotas
def test_quotas_get_defaults(self):
    quotas = self.mgr.defaults('test_project_id')
    expect = [('GET', '/v1/quotas/test_project_id/defaults', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(quotas.containers, DEFAULT_QUOTAS['containers'])
    self.assertEqual(quotas.memory, DEFAULT_QUOTAS['memory'])
    self.assertEqual(quotas.cpu, DEFAULT_QUOTAS['cpu'])
    self.assertEqual(quotas.disk, DEFAULT_QUOTAS['disk'])