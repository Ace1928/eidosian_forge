from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def mib_schema_tmplt(config_data):
    name = config_data.get('name', '')
    object_list = config_data.get('object_list', '')
    poll_interval = config_data.get('poll_interval', '')
    cmds = []
    if object_list:
        command = 'snmp-server mib bulkstat schema {name} object-list {object_list}'.format(name=name, object_list=object_list)
        cmds.append(command)
    if poll_interval:
        command = 'snmp-server mib bulkstat schema {name} poll-interval {poll_interval}'.format(name=name, poll_interval=poll_interval)
        cmds.append(command)
    if not object_list and (not poll_interval) and name:
        command = 'snmp-server mib bulkstat schema {name}'.format(name=name)
        cmds.append(command)
    return cmds