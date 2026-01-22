from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def state_merged(want, have):
    """The command generator when state is merged

    :rtype: A list
    :returns: the commands necessary to merge the provided into
              the current configuration
    """
    commands = []
    to_set = dict_diff(have, want)
    tlv_options = to_set.pop('tlv_select', {})
    for key, value in to_set.items():
        if key == 'holdtime':
            key = 'hold-time'
        if key == 'reinit':
            key = 'timer reinitialization'
        commands.append('lldp {0} {1}'.format(key, value))
    for key, value in tlv_options.items():
        device_option = key.replace('_', '-')
        if value is True:
            commands.append('lldp tlv transmit {0}'.format(device_option))
        elif value is False:
            commands.append('no lldp tlv transmit {0}'.format(device_option))
    return commands