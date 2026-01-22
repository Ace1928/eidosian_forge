from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def current_config(module, config):
    """ Parse the full port configuration for the user specified port out of
        the full configlet configuration and return as a string.

    :param module: Ansible module with parameters and client connection.
    :param config: Full config to parse specific port config from.
    :return: String of current config block for user specified port.
    """
    regex = '^interface Ethernet%s' % module.params['switch_port']
    match = re.search(regex, config, re.M)
    if not match:
        module.fail_json(msg=str('interface section not found - %s' % config))
    block_start, line_end = match.regs[0]
    match = re.search('!', config[line_end:], re.M)
    if not match:
        return config[block_start:]
    dummy, block_end = match.regs[0]
    block_end = line_end + block_end
    return config[block_start:block_end]