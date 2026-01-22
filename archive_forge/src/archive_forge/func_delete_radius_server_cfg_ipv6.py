from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
def delete_radius_server_cfg_ipv6(self, **kwargs):
    """ Delete radius server configure ipv6 """
    module = kwargs['module']
    radius_group_name = module.params['radius_group_name']
    radius_server_type = module.params['radius_server_type']
    radius_server_ipv6 = module.params['radius_server_ipv6']
    radius_server_port = module.params['radius_server_port']
    radius_server_mode = module.params['radius_server_mode']
    conf_str = CE_DELETE_RADIUS_SERVER_CFG_IPV6 % (radius_group_name, radius_server_type, radius_server_ipv6, radius_server_port, radius_server_mode)
    recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in recv_xml:
        module.fail_json(msg='Error: Create radius server config ipv6 failed.')
    cmds = []
    cmd = 'radius server group %s' % radius_group_name
    cmds.append(cmd)
    if radius_server_type == 'Authentication':
        cmd = 'undo radius server authentication %s %s' % (radius_server_ipv6, radius_server_port)
        if radius_server_mode == 'Secondary-server':
            cmd += ' secondary'
    else:
        cmd = 'undo radius server accounting  %s %s' % (radius_server_ipv6, radius_server_port)
        if radius_server_mode == 'Secondary-server':
            cmd += ' secondary'
    cmds.append(cmd)
    return cmds