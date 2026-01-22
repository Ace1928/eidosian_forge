from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def member_dicts_different(conf, member_config):
    """
    Returns if there is a difference in the replicaset configuration that we care about
    @con - The current MongoDB Replicaset configure document
    @member_config - The member dict config provided by the module. List of dicts
    """
    current_member_config = conf['members']
    member_config_defaults = {'arbiterOnly': False, 'buildIndexes': True, 'hidden': False, 'priority': {'nonarbiter': 1.0, 'arbiter': 0}, 'tags': {}, 'horizons': {}, 'secondardDelaySecs': 0, 'votes': 1}
    different = False
    msg = 'None'
    current_member_hosts = []
    for member in current_member_config:
        current_member_hosts.append(member['host'])
    member_config_hosts = []
    for member in member_config:
        if ':' not in member['host']:
            member_config_hosts.append(member['host'] + ':27017')
        else:
            member_config_hosts.append(member['host'])
    if sorted(current_member_hosts) != sorted(member_config_hosts):
        different = True
        msg = 'hosts different'
    else:
        for host in current_member_hosts:
            member_index = next((index for index, d in enumerate(current_member_config) if d['host'] == host), None)
            new_member_index = next((index for index, d in enumerate(member_config) if d['host'] == host), None)
            for config_item in member_config_defaults:
                if config_item != 'priority':
                    if current_member_config[member_index].get(config_item, member_config_defaults[config_item]) != member_config[new_member_index].get(config_item, member_config_defaults[config_item]):
                        different = True
                        msg = 'var different {0} {1} {2}'.format(config_item, current_member_config[member_index].get(config_item, member_config_defaults[config_item]), member_config[new_member_index].get(config_item, member_config_defaults[config_item]))
                        break
                else:
                    role = 'nonarbiter'
                    if current_member_config[member_index]['arbiterOnly']:
                        role = 'arbiter'
                        if current_member_config[member_index][config_item] != member_config[new_member_index].get(config_item, member_config_defaults[config_item][role]):
                            different = True
                            msg = 'var different {0}'.format(config_item)
                            break
                    elif current_member_config[member_index]['priority'] != member_config[new_member_index].get(config_item, 1.0):
                        different = True
                        msg = 'var different {0}'.format(config_item)
                        break
    return different