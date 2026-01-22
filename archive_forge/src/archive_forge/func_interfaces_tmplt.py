from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def interfaces_tmplt(config_data):
    interface = config_data.get('name', '')
    notification_linkupdown_disable = config_data.get('notification_linkupdown_disable', '')
    index_persistent = config_data.get('index_persistent', '')
    cmds = []
    if notification_linkupdown_disable:
        command = 'snmp-server interface {interface} notification linkupdown disable'.format(interface=interface)
        cmds.append(command)
    if config_data.get('index_persistent'):
        command = 'snmp-server snmp-server interface {interface} index persistence'.format(interface=interface)
        cmds.append(command)
    if not notification_linkupdown_disable and (not index_persistent) and interface:
        command = 'snmp-server interface {interface}'.format(interface=interface)
        cmds.append(command)
    return cmds