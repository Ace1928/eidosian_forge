from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
def is_pool_empty(self, poolid):
    """Check whether pool has members

        :param poolid: str - name of the pool
        :return: bool - is pool empty?
        """
    return True if not self.get_pool(poolid)['members'] else False