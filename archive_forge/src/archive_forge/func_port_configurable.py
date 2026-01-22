from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def port_configurable(module, configlet):
    """ Check configlet if the user specified port has a configuration entry
        in the configlet to determine if Ansible is allowed to configure the
        port on this switch.

    :param module: Ansible module with parameters and client connection.
    :param configlet: Dict of configlet info.
    :return: true or False.
    """
    configurable = False
    regex = '^interface Ethernet%s' % module.params['switch_port']
    for config_line in configlet['config'].split('\n'):
        if re.match(regex, config_line):
            configurable = True
    return configurable