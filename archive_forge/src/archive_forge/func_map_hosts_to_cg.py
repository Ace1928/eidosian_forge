from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def map_hosts_to_cg(self, cg_name, add_hosts):
    """Map hosts to consistency group.
            :param cg_name: The name of the consistency group
            :param add_hosts: List of hosts that are to be mapped to cg
            :return: Boolean value to indicate if hosts were mapped to cg
        """
    cg_details = self.unity_conn.get_cg(name=cg_name)
    existing_volumes_in_cg = cg_details.luns
    existing_hosts_in_cg = cg_details.block_host_access
    existing_host_ids = []
    'Get list of existing hosts in consistency group'
    if existing_hosts_in_cg:
        for i in range(len(existing_hosts_in_cg)):
            existing_host_ids.append(existing_hosts_in_cg[i].host.id)
    host_id_list = []
    host_name_list = []
    add_hosts_id = []
    host_add_list = []
    all_hosts = []
    for host in add_hosts:
        if 'host_id' in host and (not host['host_id'] in host_id_list):
            host_id_list.append(host['host_id'])
        elif 'host_name' in host and (not host['host_name'] in host_name_list):
            host_name_list.append(host['host_name'])
    'add hosts by name'
    for host_name in host_name_list:
        add_hosts_id.append(self.get_host_id_by_name(host_name))
    all_hosts = host_id_list + existing_host_ids + add_hosts_id
    add_hosts_id = list(set(all_hosts) - set(existing_host_ids))
    if len(add_hosts_id) == 0:
        return False
    if existing_volumes_in_cg:
        for host_id in add_hosts_id:
            host_dict = {'id': host_id}
            host_add_list.append(host_dict)
        LOG.info('List of hosts to be added to consistency group %s ', host_add_list)
        cg_obj = self.return_cg_instance(cg_name)
        try:
            cg_obj.modify(name=cg_name, host_add=host_add_list)
            return True
        except Exception as e:
            errormsg = 'Adding host to consistency group {0} failed with error {1}'.format(cg_name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)