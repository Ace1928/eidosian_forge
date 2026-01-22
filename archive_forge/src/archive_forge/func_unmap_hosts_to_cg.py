from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def unmap_hosts_to_cg(self, cg_name, remove_hosts):
    """Unmap hosts to consistency group.
            :param cg_name: The name of the consistency group
            :param remove_hosts: List of hosts that are to be unmapped from cg
            :return: Boolean value to indicate if hosts were mapped to cg
        """
    cg_details = self.unity_conn.get_cg(name=cg_name)
    existing_hosts_in_cg = cg_details.block_host_access
    existing_host_ids = []
    'Get host ids existing in consistency group'
    if existing_hosts_in_cg:
        for i in range(len(existing_hosts_in_cg)):
            existing_host_ids.append(existing_hosts_in_cg[i].host.id)
    host_remove_list = []
    host_id_list = []
    host_name_list = []
    remove_hosts_id = []
    for host in remove_hosts:
        if 'host_id' in host and (not host['host_id'] in host_id_list):
            host_id_list.append(host['host_id'])
        elif 'host_name' in host and (not host['host_name'] in host_name_list):
            host_name_list.append(host['host_name'])
    'remove hosts by name'
    for host in host_name_list:
        remove_hosts_id.append(self.get_host_id_by_name(host))
    host_id_list = list(set(host_id_list + remove_hosts_id))
    remove_hosts_id = list(set(existing_host_ids).intersection(set(host_id_list)))
    if len(remove_hosts_id) == 0:
        return False
    for host in remove_hosts_id:
        host_dict = {'id': host}
        host_remove_list.append(host_dict)
    cg_obj = self.return_cg_instance(cg_name)
    try:
        cg_obj.modify(name=cg_name, host_remove=host_remove_list)
        return True
    except Exception as e:
        errormsg = 'Removing host from consistency group {0} failed with error {1}'.format(cg_name, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)