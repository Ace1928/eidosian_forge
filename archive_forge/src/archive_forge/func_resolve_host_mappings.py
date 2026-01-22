from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def resolve_host_mappings(self, hosts):
    """ This method creates a dictionary of hosts and hlu parameter values
            :param hosts: host and hlu value passed from input file
            :return: list of host and hlu dictionary
        """
    host_list_new = []
    if hosts:
        for item in hosts:
            host_dict = dict()
            host_id = None
            hlu = None
            if item['host_name']:
                host = self.get_host(host_name=item['host_name'])
                if host:
                    host_id = host.id
            if item['host_id']:
                host_id = item['host_id']
            if item['hlu']:
                hlu = item['hlu']
            host_dict['host_id'] = host_id
            host_dict['hlu'] = hlu
            host_list_new.append(host_dict)
    return host_list_new