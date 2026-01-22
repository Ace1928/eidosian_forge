from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def update_global_pool(self, config):
    """
        Update/Create Global Pool in Cisco Catalyst Center with fields provided in playbook

        Parameters:
            config (list of dict) - Playbook details

        Returns:
            None
        """
    name = config.get('global_pool_details').get('settings').get('ip_pool')[0].get('name')
    result_global_pool = self.result.get('response')[0].get('globalPool')
    result_global_pool.get('response').update({name: {}})
    if not self.have.get('globalPool').get('exists'):
        pool_params = self.want.get('wantGlobal')
        self.log('Desired State for global pool (want): {0}'.format(pool_params), 'DEBUG')
        response = self.dnac._exec(family='network_settings', function='create_global_pool', params=pool_params)
        self.check_execution_response_status(response).check_return_status()
        self.log("Successfully created global pool '{0}'.".format(name), 'INFO')
        result_global_pool.get('response').get(name).update({'globalPool Details': self.want.get('wantGlobal')})
        result_global_pool.get('msg').update({name: 'Global Pool Created Successfully'})
        return
    if not self.requires_update(self.have.get('globalPool').get('details'), self.want.get('wantGlobal'), self.global_pool_obj_params):
        self.log("Global pool '{0}' doesn't require an update".format(name), 'INFO')
        result_global_pool.get('response').get(name).update({'Cisco Catalyst Center params': self.have.get('globalPool').get('details').get('settings').get('ippool')[0]})
        result_global_pool.get('response').get(name).update({'Id': self.have.get('globalPool').get('id')})
        result_global_pool.get('msg').update({name: "Global pool doesn't require an update"})
        return
    self.log('Global pool requires update', 'DEBUG')
    pool_params = copy.deepcopy(self.want.get('wantGlobal'))
    pool_params_ippool = pool_params.get('settings').get('ippool')[0]
    pool_params_ippool.update({'id': self.have.get('globalPool').get('id')})
    self.log('Desired State for global pool (want): {0}'.format(pool_params), 'DEBUG')
    keys_to_remove = ['IpAddressSpace', 'ipPoolCidr', 'type']
    for key in keys_to_remove:
        del pool_params['settings']['ippool'][0][key]
    have_ippool = self.have.get('globalPool').get('details').get('settings').get('ippool')[0]
    keys_to_update = ['dhcpServerIps', 'dnsServerIps', 'gateway']
    for key in keys_to_update:
        if pool_params_ippool.get(key) is None:
            pool_params_ippool[key] = have_ippool.get(key)
    self.log('Desired global pool details (want): {0}'.format(pool_params), 'DEBUG')
    response = self.dnac._exec(family='network_settings', function='update_global_pool', params=pool_params)
    self.check_execution_response_status(response).check_return_status()
    self.log("Global pool '{0}' updated successfully".format(name), 'INFO')
    result_global_pool.get('response').get(name).update({'Id': self.have.get('globalPool').get('details').get('id')})
    result_global_pool.get('msg').update({name: 'Global Pool Updated Successfully'})
    return