from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def update_reserve_pool(self, config):
    """
        Update or Create a Reserve Pool in Cisco Catalyst Center based on the provided configuration.
        This method checks if a reserve pool with the specified name exists in Cisco Catalyst Center.
        If it exists and requires an update, it updates the pool. If not, it creates a new pool.

        Parameters:
            config (list of dict) - Playbook details containing Reserve Pool information.

        Returns:
            None
        """
    name = config.get('reserve_pool_details').get('name')
    result_reserve_pool = self.result.get('response')[1].get('reservePool')
    result_reserve_pool.get('response').update({name: {}})
    self.log('Current reserved pool details in Catalyst Center: {0}'.format(self.have.get('reservePool').get('details')), 'DEBUG')
    self.log('Desired reserved pool details in Catalyst Center: {0}'.format(self.want.get('wantReserve')), 'DEBUG')
    self.log('IPv4 global pool: {0}'.format(self.want.get('wantReserve').get('ipv4GlobalPool')), 'DEBUG')
    site_name = config.get('reserve_pool_details').get('site_name')
    reserve_params = self.want.get('wantReserve')
    site_id = self.get_site_id(site_name)
    reserve_params.update({'site_id': site_id})
    if not self.have.get('reservePool').get('exists'):
        self.log('Desired reserved pool details (want): {0}'.format(reserve_params), 'DEBUG')
        response = self.dnac._exec(family='network_settings', function='reserve_ip_subpool', params=reserve_params)
        self.check_execution_response_status(response).check_return_status()
        self.log("Successfully created IP subpool reservation '{0}'.".format(name), 'INFO')
        result_reserve_pool.get('response').get(name).update({'reservePool Details': self.want.get('wantReserve')})
        result_reserve_pool.get('msg').update({name: 'Ip Subpool Reservation Created Successfully'})
        return
    if not self.requires_update(self.have.get('reservePool').get('details'), self.want.get('wantReserve'), self.reserve_pool_obj_params):
        self.log("Reserved ip subpool '{0}' doesn't require an update".format(name), 'INFO')
        result_reserve_pool.get('response').get(name).update({'Cisco Catalyst Center params': self.have.get('reservePool').get('details')})
        result_reserve_pool.get('response').get(name).update({'Id': self.have.get('reservePool').get('id')})
        result_reserve_pool.get('msg').update({name: "Reserve ip subpool doesn't require an update"})
        return
    self.log("Reserved ip pool '{0}' requires an update".format(name), 'DEBUG')
    self.log("Current reserved ip pool '{0}' details in Catalyst Center: {1}".format(name, self.have.get('reservePool')), 'DEBUG')
    self.log("Desired reserved ip pool '{0}' details: {1}".format(name, self.want.get('wantReserve')), 'DEBUG')
    reserve_params.update({'id': self.have.get('reservePool').get('id')})
    response = self.dnac._exec(family='network_settings', function='update_reserve_ip_subpool', params=reserve_params)
    self.check_execution_response_status(response).check_return_status()
    self.log("Reserved ip subpool '{0}' updated successfully.".format(name), 'INFO')
    result_reserve_pool['msg'] = 'Reserved Ip Subpool Updated Successfully'
    result_reserve_pool.get('response').get(name).update({'Reservation details': self.have.get('reservePool').get('details')})
    return