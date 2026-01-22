from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def merge_radius_template(self, **kwargs):
    """ Merge radius template """
    radius_server_group = kwargs['radius_server_group']
    module = kwargs['module']
    conf_str = CE_MERGE_RADIUS_TEMPLATE % radius_server_group
    xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in xml:
        module.fail_json(msg='Error: Merge radius template failed.')
    cmds = []
    cmd = 'radius server group %s' % radius_server_group
    cmds.append(cmd)
    return cmds