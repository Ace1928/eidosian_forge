from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def host_access_modify_required(self, host_access_list):
    """Check if host access modification is required
            :param host_access_list: host access dict list
            :return: Dict with attributes to modify, or None if no
            modification is required.
        """
    try:
        to_modify = False
        mapping_state = self.module.params['mapping_state']
        host_id_list = []
        hlu_list = []
        new_list = []
        if not host_access_list and self.new_host_list and (mapping_state == 'unmapped'):
            return to_modify
        elif host_access_list:
            for host_access in host_access_list.host:
                host_id_list.append(host_access.id)
                host = self.get_host(host_id=host_access.id).update()
                host_dict = host.host_luns._get_properties()
                LOG.debug('check if hlu present : %s', host_dict)
                if 'hlu' in host_dict.keys():
                    hlu_list.append(host_dict['hlu'])
        if mapping_state == 'mapped':
            if self.param_host_id not in host_id_list:
                for item in self.new_host_list:
                    new_list.append(item.get('host_id'))
                if not list(set(new_list) - set(host_id_list)):
                    return False
                to_modify = True
        if mapping_state == 'unmapped':
            if self.new_host_list:
                for item in self.new_host_list:
                    new_list.append(item.get('host_id'))
                if list(set(new_list) - set(host_id_list)):
                    return False
                self.overlapping_list = list(set(host_id_list) - set(new_list))
                to_modify = True
        LOG.debug('host_access_modify_required : %s ', str(to_modify))
        return to_modify
    except Exception as e:
        errormsg = 'Failed to compare the host_access with error {0} {1}'.format(host_access_list, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)