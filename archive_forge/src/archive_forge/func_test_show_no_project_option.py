from openstack.tests.functional import base
def test_show_no_project_option(self):
    top = self.operator_cloud.network.get_auto_allocated_topology()
    project = self.conn.session.get_project_id()
    network = self.operator_cloud.network.get_network(top.id)
    self.assertEqual(top.project_id, project)
    self.assertEqual(top.id, network.id)