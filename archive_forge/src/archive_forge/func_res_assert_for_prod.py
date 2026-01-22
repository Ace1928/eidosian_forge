from heat_integrationtests.functional import functional_base
def res_assert_for_prod(self, stack_identifier, bj_prod=True, fj_zone=False, shannxi_provice=False):

    def is_not_deleted(r):
        return r.resource_status != 'DELETE_COMPLETE'
    resources = self.list_resources(stack_identifier, is_not_deleted)
    res_names = set(resources)
    if bj_prod:
        self.assertEqual(4, len(resources))
        self.assertIn('beijing_prod_res', res_names)
        self.assertIn('not_shannxi_res', res_names)
    elif fj_zone:
        self.assertEqual(5, len(resources))
        self.assertIn('fujian_res', res_names)
        self.assertNotIn('beijing_prod_res', res_names)
        self.assertIn('not_shannxi_res', res_names)
        self.assertIn('fujian_prod_res', res_names)
    elif shannxi_provice:
        self.assertEqual(3, len(resources))
        self.assertIn('shannxi_res', res_names)
    else:
        self.assertEqual(3, len(resources))
        self.assertIn('not_shannxi_res', res_names)
    self.assertIn('prod_res', res_names)
    self.assertIn('test_res', res_names)