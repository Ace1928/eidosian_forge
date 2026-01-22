from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def update_network_offering(self, network_offering):
    tags = self.module.params.get('tags')
    domains = self.module.params.get('domains')
    zones = self.module.params.get('zones')
    args = {'id': network_offering['id'], 'state': self.module.params.get('state'), 'displaytext': self.module.params.get('display_text'), 'name': self.module.params.get('name'), 'availability': self.module.params.get('availability'), 'maxconnections': self.module.params.get('max_connections'), 'tags': ','.join(tags) if tags else None, 'domainid': ','.join(domains) if domains else None, 'zoneid': ','.join(zones) if zones else None}
    if args['state'] in ['enabled', 'disabled']:
        args['state'] = args['state'].title()
    else:
        del args['state']
    if self.has_changed(args, network_offering):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('updateNetworkOffering', **args)
            network_offering = res['networkoffering']
    return network_offering