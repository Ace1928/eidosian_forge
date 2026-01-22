from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def get_host_count(self, host_name):
    """ To get the count of hosts with same host_name """
    hosts = []
    host_count = 0
    hosts = utils.host.UnityHostList.get(cli=self.unity._cli, name=host_name)
    host_count = len(hosts)
    return host_count