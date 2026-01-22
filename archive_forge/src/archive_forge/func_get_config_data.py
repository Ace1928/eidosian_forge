from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def get_config_data(self):
    """get configuration commands and current configuration"""
    self.exist_conf['type'] = self.type
    self.exist_conf['source_ip'] = None
    self.exist_conf['host_ip'] = None
    self.exist_conf['host_port'] = None
    self.exist_conf['host_vpn'] = None
    self.exist_conf['version'] = None
    self.exist_conf['as_option'] = None
    self.exist_conf['bgp_netxhop'] = 'disable'
    self.config = self.get_netstream_config()
    if self.type and self.source_ip:
        self.config_nets_export_src_addr()
    if self.type and self.host_ip and self.host_port:
        self.config_nets_export_host_addr()
    if self.type == 'vxlan' and self.version == '9':
        self.config_nets_export_vxlan_ver()
    if self.type == 'ip' and self.version:
        self.config_nets_export_ip_ver()