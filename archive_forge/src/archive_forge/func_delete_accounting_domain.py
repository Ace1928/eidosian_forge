from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_accounting_domain(self, **kwargs):
    """ Delete domain of accounting """
    domain_name = kwargs['domain_name']
    acct_scheme_name = kwargs['acct_scheme_name']
    module = kwargs['module']
    if domain_name == 'default':
        return SUCCESS
    conf_str = CE_DELETE_ACCOUNTING_DOMAIN % (domain_name, acct_scheme_name)
    xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in xml:
        module.fail_json(msg='Error: Delete accounting domain failed.')
    cmds = []
    cmd = 'undo domain %s' % domain_name
    cmds.append(cmd)
    cmd = 'undo accounting-scheme'
    cmds.append(cmd)
    return cmds