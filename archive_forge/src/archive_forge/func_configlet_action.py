from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def configlet_action(module, configlet):
    """ Take appropriate action based on current state of device and user
        requested action.

        Return current config block for specified port if action is show.

        If action is add or remove make the appropriate changes to the
        configlet and return the associated information.

    :param module: Ansible module with parameters and client connection.
    :param configlet: Dict of configlet info.
    :return: Dict of information to updated results with.
    """
    result = dict()
    existing_config = current_config(module, configlet['config'])
    if module.params['action'] == 'show':
        result['currentConfigBlock'] = existing_config
        return result
    elif module.params['action'] == 'add':
        result['newConfigBlock'] = config_from_template(module)
    elif module.params['action'] == 'remove':
        result['newConfigBlock'] = 'interface Ethernet%s\n!' % module.params['switch_port']
    result['oldConfigBlock'] = existing_config
    result['fullConfig'] = updated_configlet_content(module, configlet['config'], result['newConfigBlock'])
    resp = module.client.api.update_configlet(result['fullConfig'], configlet['key'], configlet['name'])
    if 'data' in resp:
        result['updateConfigletResponse'] = resp['data']
        if 'task' in resp['data']:
            result['changed'] = True
            result['taskCreated'] = True
    return result