from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, exec_command, run_commands
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import ComplexList
def get_rollback_dict(self):
    """ get rollback attributes dict."""
    rollback_info = dict()
    rollback_info['RollBackInfos'] = list()
    flags = list()
    exp = 'commit list'
    flags.append(exp)
    cfg_info = self.get_config(flags)
    if not cfg_info:
        return rollback_info
    cfg_line = cfg_info.split('\n')
    for cfg in cfg_line:
        if re.findall('^\\d', cfg):
            pre_rollback_info = cfg.split()
            rollback_info['RollBackInfos'].append(dict(commitId=pre_rollback_info[1].replace('*', ''), userLabel=pre_rollback_info[2]))
    return rollback_info