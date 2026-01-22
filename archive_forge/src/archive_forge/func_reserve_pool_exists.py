from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def reserve_pool_exists(self, name, site_name):
    """
        Check if the Reserved pool with the given name exists in a specific site
        Use check_return_status() to check for failure

        Parameters:
            name (str) - The name of the Reserved pool to check for existence.
            site_name (str) - The name of the site where the Reserved pool is located.

        Returns:
            dict - A dictionary containing information about the Reserved pool's existence:
            - 'exists' (bool): True if the Reserved pool exists in the specified site, else False.
            - 'id' (str or None): The ID of the Reserved pool if it exists, or None if it doesn't.
            - 'details' (dict or None): Details of the Reserved pool if it exists, or else None.
        """
    reserve_pool = {'exists': False, 'details': None, 'id': None, 'success': True}
    site_id = self.get_site_id(site_name)
    self.log('Site ID for the site name {0}: {1}'.format(site_name, site_id), 'DEBUG')
    if not site_id:
        reserve_pool.update({'success': False})
        self.msg = 'Failed to get the site id from the site name {0}'.format(site_name)
        self.status = 'failed'
        return reserve_pool
    response = self.dnac._exec(family='network_settings', function='get_reserve_ip_subpool', params={'siteId': site_id})
    if not isinstance(response, dict):
        reserve_pool.update({'success': False})
        self.msg = 'Error in getting reserve pool - Response is not a dictionary'
        self.status = 'exited'
        return reserve_pool
    all_reserve_pool_details = response.get('response')
    reserve_pool_details = get_dict_result(all_reserve_pool_details, 'groupName', name)
    if not reserve_pool_details:
        self.log('Reserved pool {0} does not exist in the site {1}'.format(name, site_name), 'DEBUG')
        return reserve_pool
    reserve_pool.update({'exists': True})
    reserve_pool.update({'id': reserve_pool_details.get('id')})
    reserve_pool.update({'details': self.get_reserve_pool_params(reserve_pool_details)})
    self.log('Reserved pool details: {0}'.format(reserve_pool.get('details')), 'DEBUG')
    self.log('Reserved pool id: {0}'.format(reserve_pool.get('id')), 'DEBUG')
    return reserve_pool