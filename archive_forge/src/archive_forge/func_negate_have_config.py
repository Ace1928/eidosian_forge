from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import dict_to_set
def negate_have_config(want_diff, have_diff, vlan, commands):
    name = dict(have_diff).get('name')
    if name and (not dict(want_diff).get('name')):
        self.remove_command_from_config_list(vlan, 'name {0}'.format(name), commands)
    state = dict(have_diff).get('state')
    if state and (not dict(want_diff).get('state')):
        self.remove_command_from_config_list(vlan, 'state {0}'.format(state), commands)
    shutdown = dict(have_diff).get('shutdown')
    if shutdown and (not dict(want_diff).get('shutdown')):
        self.remove_command_from_config_list(vlan, 'shutdown', commands)
    mtu = dict(have_diff).get('mtu')
    if mtu and (not dict(want_diff).get('mtu')):
        self.remove_command_from_config_list(vlan, 'mtu {0}'.format(mtu), commands)
    remote_span = dict(have_diff).get('remote_span')
    if remote_span and (not dict(want_diff).get('remote_span')):
        self.remove_command_from_config_list(vlan, 'remote-span', commands)
    private_vlan = dict(have_diff).get('private_vlan')
    if private_vlan and (not dict(want_diff).get('private_vlan')):
        private_vlan_type = dict(private_vlan).get('type')
        self.remove_command_from_config_list(vlan, 'private-vlan {0}'.format(private_vlan_type), commands)
        if private_vlan_type == 'primary' and dict(private_vlan).get('associated'):
            self.remove_command_from_config_list(vlan, 'private-vlan association', commands)