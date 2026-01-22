from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def check_ip_in_standby(self, standby_ip, cluster_details):
    """whether standby IPs present in standby MDMs"""
    if 'standbyMDMs' in cluster_details:
        for stb_mdm in cluster_details['standbyMDMs']:
            current_stb_ips = stb_mdm['ips']
            for ips in standby_ip:
                if ips in current_stb_ips:
                    LOG.info(self.exist_msg)
                    return False
    return True