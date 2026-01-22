from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def gather_host_portgroup_info(self):
    """Gather Port Group info per ESXi host"""
    hosts_pg_info = dict()
    for host in self.hosts:
        pgs = host.config.network.portgroup
        hosts_pg_info[host.name] = []
        for portgroup in pgs:
            hosts_pg_info[host.name].append(self.normalize_pg_info(portgroup_obj=portgroup, policy_info=self.policies))
    return hosts_pg_info