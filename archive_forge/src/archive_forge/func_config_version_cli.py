from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, \
def config_version_cli(self):
    """ Config version by cli """
    if 'disable' in self.cur_cli_cfg['version']:
        cmd = 'snmp-agent sys-info version %s' % self.version
        self.updates_cmd.append(cmd)
        cmds = list()
        cmds.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True
    elif self.version != self.cur_cli_cfg['version']:
        cmd = 'snmp-agent sys-info version  %s disable' % self.cur_cli_cfg['version']
        self.updates_cmd.append(cmd)
        cmd = 'snmp-agent sys-info version  %s' % self.version
        self.updates_cmd.append(cmd)
        cmds = list()
        cmds.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True