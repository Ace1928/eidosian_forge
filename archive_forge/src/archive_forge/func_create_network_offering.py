from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def create_network_offering(self):
    network_offering = None
    self.result['changed'] = True
    args = {'state': self.module.params.get('state'), 'displaytext': self.module.params.get('display_text'), 'guestiptype': self.module.params.get('guest_ip_type'), 'name': self.module.params.get('name'), 'supportedservices': self.module.params.get('supported_services'), 'traffictype': self.module.params.get('traffic_type'), 'availability': self.module.params.get('availability'), 'conservemode': self.module.params.get('conserve_mode'), 'details': self.module.params.get('details'), 'egressdefaultpolicy': self.module.params.get('egress_default_policy') == 'allow', 'ispersistent': self.module.params.get('persistent'), 'keepaliveenabled': self.module.params.get('keepalive_enabled'), 'maxconnections': self.module.params.get('max_connections'), 'networkrate': self.module.params.get('network_rate'), 'servicecapabilitylist': self.module.params.get('service_capabilities'), 'serviceofferingid': self.get_service_offering_id(), 'serviceproviderlist': self.module.params.get('service_providers'), 'specifyipranges': self.module.params.get('specify_ip_ranges'), 'specifyvlan': self.module.params.get('specify_vlan'), 'forvpc': self.module.params.get('for_vpc'), 'tags': self.module.params.get('tags'), 'domainid': self.module.params.get('domains'), 'zoneid': self.module.params.get('zones')}
    required_params = ['display_text', 'guest_ip_type', 'supported_services', 'service_providers']
    self.module.fail_on_missing_params(required_params=required_params)
    if not self.module.check_mode:
        res = self.query_api('createNetworkOffering', **args)
        network_offering = res['networkoffering']
    return network_offering