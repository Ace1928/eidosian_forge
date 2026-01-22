from openstack.tests.functional import base
def test_dry_run_option_pass(self):
    networks = self.operator_cloud.network.networks()
    self._set_network_external(networks)
    top = self.operator_cloud.network.validate_auto_allocated_topology(self.PROJECT_ID)
    self.assertEqual(self.PROJECT_ID, top.project)
    self.assertEqual('dry-run=pass', top.id)