from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_fake_hardware_list_details(self):
    drivers = self.conn.baremetal.drivers(details=True)
    driver = [d for d in drivers if d.name == 'fake-hardware'][0]
    for iface in ('boot', 'deploy', 'management', 'power'):
        self.assertIn('fake', getattr(driver, 'enabled_%s_interfaces' % iface))
        self.assertEqual('fake', getattr(driver, 'default_%s_interface' % iface))