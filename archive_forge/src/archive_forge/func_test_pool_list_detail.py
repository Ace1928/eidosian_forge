from manilaclient.tests.functional.osc import base
def test_pool_list_detail(self):
    pools = self.list_pools(detail=True)
    self.assertTableStruct(pools, ['Name', 'Host', 'Backend', 'Pool', 'Capabilities'])
    self.assertTrue(len(pools) > 0)