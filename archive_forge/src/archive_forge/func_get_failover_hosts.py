from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils._text import to_native
def get_failover_hosts(self):
    """
        Get failover hosts for failover_host_admission_control policy
        Returns: List of ESXi hosts sorted by name

        """
    policy = self.params.get('failover_host_admission_control')
    hosts = []
    all_hosts = dict(((h.name, h) for h in self.get_all_hosts_by_cluster(self.cluster_name)))
    for host in policy.get('failover_hosts'):
        if host in all_hosts:
            hosts.append(all_hosts.get(host))
        else:
            self.module.fail_json(msg='Host %s is not a member of cluster %s.' % (host, self.cluster_name))
    hosts.sort(key=lambda h: h.name)
    return hosts