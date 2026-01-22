from heat_integrationtests.functional import functional_base
def test_deployment_crud_with_rolling_update(self):
    self.deployment_crud(self.sd_template_with_upd_policy)