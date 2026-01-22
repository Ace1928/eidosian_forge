from __future__ import (absolute_import, division, print_function)
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, set_nc_config, get_nc_config, execute_nc_action
def config_global_dldp(self):
    """Config global dldp"""
    if self.same_conf:
        return
    enable = self.enable
    if not self.enable:
        enable = self.dldp_conf['dldpEnable']
    if enable == 'enable':
        enable = 'true'
    else:
        enable = 'false'
    internal = self.internal
    if not self.internal:
        internal = self.dldp_conf['dldpInterval']
    work_mode = self.work_mode
    if not self.work_mode:
        work_mode = self.dldp_conf['dldpWorkMode']
    if work_mode == 'enhance' or work_mode == 'dldpEnhance':
        work_mode = 'dldpEnhance'
    else:
        work_mode = 'dldpNormal'
    auth_mode = self.auth_mode
    if not self.auth_mode:
        auth_mode = self.dldp_conf['dldpAuthMode']
    if auth_mode == 'md5':
        auth_mode = 'dldpAuthMD5'
    elif auth_mode == 'simple':
        auth_mode = 'dldpAuthSimple'
    elif auth_mode == 'sha':
        auth_mode = 'dldpAuthSHA'
    elif auth_mode == 'hmac-sha256':
        auth_mode = 'dldpAuthHMAC-SHA256'
    elif auth_mode == 'none':
        auth_mode = 'dldpAuthNone'
    xml_str = CE_NC_MERGE_DLDP_GLOBAL_CONFIG_HEAD % (enable, internal, work_mode)
    if self.auth_mode:
        if self.auth_mode == 'none':
            xml_str += '<dldpAuthMode>dldpAuthNone</dldpAuthMode>'
        else:
            xml_str += '<dldpAuthMode>%s</dldpAuthMode>' % auth_mode
            xml_str += '<dldpPasswords>%s</dldpPasswords>' % self.auth_pwd
    xml_str += CE_NC_MERGE_DLDP_GLOBAL_CONFIG_TAIL
    ret_xml = set_nc_config(self.module, xml_str)
    self.check_response(ret_xml, 'MERGE_DLDP_GLOBAL_CONFIG')
    if self.reset == 'enable':
        xml_str = CE_NC_ACTION_RESET_DLDP
        ret_xml = execute_nc_action(self.module, xml_str)
        self.check_response(ret_xml, 'ACTION_RESET_DLDP')
    self.changed = True