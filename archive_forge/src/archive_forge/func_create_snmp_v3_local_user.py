from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def create_snmp_v3_local_user(self, **kwargs):
    """ Create snmp v3 local user operation """
    module = kwargs['module']
    local_user_name = module.params['aaa_local_user']
    auth_protocol = module.params['auth_protocol']
    auth_key = module.params['auth_key']
    priv_protocol = module.params['priv_protocol']
    priv_key = module.params['priv_key']
    conf_str = CE_CREATE_SNMP_V3_LOCAL_USER % (local_user_name, auth_protocol, auth_key, priv_protocol, priv_key)
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Create snmp v3 local user failed.')
    cmd = 'snmp-agent local-user v3 %s ' % local_user_name + 'authentication-mode %s ' % auth_protocol + 'cipher ****** ' + 'privacy-mode %s ' % priv_protocol + 'cipher  ******'
    return cmd