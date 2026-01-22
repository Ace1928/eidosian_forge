import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_flavor_add_remove_profile(self):
    """Test add and remove network flavor to/from profile"""
    name1 = uuid.uuid4().hex
    cmd_output1 = self.openstack('network flavor create --description testdescription --enable  --service-type L3_ROUTER_NAT ' + name1, parse_output=True)
    flavor_id = cmd_output1.get('id')
    cmd_output2 = self.openstack('network flavor profile create --description fakedescription --enable --metainfo Extrainfo', parse_output=True)
    service_profile_id = cmd_output2.get('id')
    self.addCleanup(self.openstack, 'network flavor delete %s' % flavor_id)
    self.addCleanup(self.openstack, 'network flavor profile delete %s' % service_profile_id)
    self.openstack('network flavor add profile ' + flavor_id + ' ' + service_profile_id)
    cmd_output4 = self.openstack('network flavor show ' + flavor_id, parse_output=True)
    service_profile_ids1 = cmd_output4.get('service_profile_ids')
    self.assertIn(service_profile_id, service_profile_ids1)
    self.openstack('network flavor remove profile ' + flavor_id + ' ' + service_profile_id)
    cmd_output6 = self.openstack('network flavor show ' + flavor_id, parse_output=True)
    service_profile_ids2 = cmd_output6.get('service_profile_ids')
    self.assertNotIn(service_profile_id, service_profile_ids2)