from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
def is_pool_existing(self, poolid):
    """Check whether pool already exist

        :param poolid: str - name of the pool
        :return: bool - is pool exists?
        """
    try:
        pools = self.proxmox_api.pools.get()
        for pool in pools:
            if pool['poolid'] == poolid:
                return True
        return False
    except Exception as e:
        self.module.fail_json(msg='Unable to retrieve pools: {0}'.format(e))