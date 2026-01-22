from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def set_vrrp_global(self):
    """set vrrp global attribute info"""
    if self.is_vrrp_global_info_change():
        conf_str = CE_NC_SET_VRRP_GLOBAL_HEAD
        if self.gratuitous_arp_interval:
            conf_str += '<gratuitousArpTimeOut>%s</gratuitousArpTimeOut>' % self.gratuitous_arp_interval
        if self.recover_delay:
            conf_str += '<recoverDelay>%s</recoverDelay>' % self.recover_delay
        if self.version:
            conf_str += '<version>%s</version>' % self.version
        conf_str += CE_NC_SET_VRRP_GLOBAL_TAIL
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: set vrrp global attribute info failed.')
        if self.gratuitous_arp_interval:
            self.updates_cmd.append('vrrp gratuitous-arp interval %s' % self.gratuitous_arp_interval)
        if self.recover_delay:
            self.updates_cmd.append('vrrp recover-delay %s' % self.recover_delay)
        if self.version:
            version = '3'
            if self.version == 'v2':
                version = '2'
            self.updates_cmd.append('vrrp version %s' % version)
        self.changed = True