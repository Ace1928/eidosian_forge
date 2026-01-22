from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_local_user_group(self, **kwargs):
    """ Delete local user group """
    local_user_group = kwargs['local_user_group']
    module = kwargs['module']
    conf_str = CE_DELETE_LOCAL_USER_GROUP % local_user_group
    xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in xml:
        module.fail_json(msg='Error: Delete local user group failed.')
    cmds = []
    cmd = 'undo user-group %s' % local_user_group
    cmds.append(cmd)
    return cmds