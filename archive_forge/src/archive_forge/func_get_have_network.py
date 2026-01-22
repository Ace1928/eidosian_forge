from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_have_network(self, config):
    """
        Get the current Network details from Cisco Catalyst
        Center based on the provided playbook details.

        Parameters:
            config (dict) - Playbook details containing Network Management configuration.

        Returns:
            self - The current object with updated Network information.
        """
    network = {}
    site_name = config.get('network_management_details').get('site_name')
    if site_name is None:
        self.msg = "Mandatory Parameter 'site_name' missing"
        self.status = 'failed'
        return self
    site_id = self.get_site_id(site_name)
    if site_id is None:
        self.msg = 'Failed to get site id from {0}'.format(site_name)
        self.status = 'failed'
        return self
    network['site_id'] = site_id
    network['net_details'] = self.get_network_params(site_id)
    self.log('Network details from the Catalyst Center: {0}'.format(network), 'DEBUG')
    self.have.update({'network': network})
    self.msg = 'Collecting the network details from the Cisco Catalyst Center'
    self.status = 'success'
    return self