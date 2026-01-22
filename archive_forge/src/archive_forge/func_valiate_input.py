from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
def valiate_input(playvals, type, module):
    """
    Helper method to validate playbook values for destination groups
    """
    if type == 'destination_groups':
        if not playvals.get('id'):
            msg = 'Invalid playbook value: {0}.'.format(playvals)
            msg += ' Parameter <id> under <destination_groups> is required'
            module.fail_json(msg=msg)
        if playvals.get('destination') and (not isinstance(playvals['destination'], dict)):
            msg = 'Invalid playbook value: {0}.'.format(playvals)
            msg += ' Parameter <destination> under <destination_groups> must be a dict'
            module.fail_json(msg=msg)
        if not playvals.get('destination') and len(playvals) > 1:
            msg = 'Invalid playbook value: {0}.'.format(playvals)
            msg += ' Playbook entry contains unrecongnized parameters.'
            msg += ' Make sure <destination> keys under <destination_groups> are specified as follows:'
            msg += ' destination: {ip: <ip>, port: <port>, protocol: <prot>, encoding: <enc>}}'
            module.fail_json(msg=msg)
    if type == 'sensor_groups':
        if not playvals.get('id'):
            msg = 'Invalid playbook value: {0}.'.format(playvals)
            msg += ' Parameter <id> under <sensor_groups> is required'
            module.fail_json(msg=msg)
        if playvals.get('path') and 'name' not in playvals['path'].keys():
            msg = 'Invalid playbook value: {0}.'.format(playvals)
            msg += ' Parameter <path> under <sensor_groups> requires <name> key'
            module.fail_json(msg=msg)