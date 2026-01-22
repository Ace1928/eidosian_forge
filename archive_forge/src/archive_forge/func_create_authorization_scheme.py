from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def create_authorization_scheme(self, **kwargs):
    """ Create scheme of authorization """
    author_scheme_name = kwargs['author_scheme_name']
    first_author_mode = kwargs['first_author_mode']
    module = kwargs['module']
    conf_str = CE_CREATE_AUTHORIZATION_SCHEME % (author_scheme_name, first_author_mode)
    xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in xml:
        module.fail_json(msg='Error: Create authorization scheme failed.')
    cmds = []
    cmd = 'authorization-scheme %s' % author_scheme_name
    cmds.append(cmd)
    cmd = 'authorization-mode %s' % first_author_mode
    cmds.append(cmd)
    return cmds