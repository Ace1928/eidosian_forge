from cinderclient.v3 import availability_zones
from cinderclient.v3 import shell
from cinderclient.tests.unit.fixture_data import availability_zones as azfixture  # noqa
from cinderclient.tests.unit.fixture_data import client
from cinderclient.tests.unit import utils
def test_detail_availability_zone(self):
    zones = self.cs.availability_zones.list(detailed=True)
    self.assert_called('GET', '/os-availability-zone/detail')
    self._assert_request_id(zones)
    for zone in zones:
        self.assertIsInstance(zone, availability_zones.AvailabilityZone)
    self.assertEqual(3, len(zones))
    l0 = ['zone-1', 'available']
    l1 = ['|- fake_host-1', '']
    l2 = ['| |- cinder-volume', 'enabled :-) 2012-12-26 14:45:25']
    l3 = ['internal', 'available']
    l4 = ['|- fake_host-1', '']
    l5 = ['| |- cinder-sched', 'enabled :-) 2012-12-26 14:45:24']
    l6 = ['zone-2', 'not available']
    z0 = shell.treeizeAvailabilityZone(zones[0])
    z1 = shell.treeizeAvailabilityZone(zones[1])
    z2 = shell.treeizeAvailabilityZone(zones[2])
    self.assertEqual((3, 3, 1), (len(z0), len(z1), len(z2)))
    self._assertZone(z0[0], l0[0], l0[1])
    self._assertZone(z0[1], l1[0], l1[1])
    self._assertZone(z0[2], l2[0], l2[1])
    self._assertZone(z1[0], l3[0], l3[1])
    self._assertZone(z1[1], l4[0], l4[1])
    self._assertZone(z1[2], l5[0], l5[1])
    self._assertZone(z2[0], l6[0], l6[1])