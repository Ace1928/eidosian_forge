from openstack.tests.functional import base
def test_get_find_hypervisors(self):
    for hypervisor in self.conn.compute.hypervisors():
        self.conn.compute.get_hypervisor(hypervisor.id)
        self.conn.compute.find_hypervisor(hypervisor.id)