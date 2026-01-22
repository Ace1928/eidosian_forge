from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def mib_bulkstat_transfer_ids_tmplt(config_data):
    name = config_data.get('name', '')
    buffer_size = config_data.get('buffer_size', '')
    enable = config_data.get('enable', '')
    format_schemaASCI = config_data.get('format_schemaASCI', '')
    retain = config_data.get('retain', '')
    retry = config_data.get('retry', '')
    schema = config_data.get('schema', '')
    transfer_interval = config_data.get('transfer_interval', '')
    cmds = []
    if buffer_size:
        command = 'snmp-server mib bulkstat transfer-id {name} buffer-size {buffer_size}'.format(name=name, buffer_size=buffer_size)
        cmds.append(command)
    if enable:
        command = 'snmp-server mib bulkstat transfer-id {name} enable'.format(name=name)
        cmds.append(command)
    if format_schemaASCI:
        command = 'snmp-server mib bulkstat transfer-id {name} format schemaASCII'.format(name=name)
        cmds.append(command)
    if retain:
        command = 'snmp-server mib bulkstat transfer-id {name} retain {retain}'.format(name=name, retain=retain)
        cmds.append(command)
    if retry:
        command = 'snmp-server mib bulkstat transfer-id {name} retry {retry}'.format(name=name, retry=retry)
        cmds.append(command)
    if schema:
        command = 'snmp-server mib bulkstat transfer-id {name} schema {schema}'.format(name=name, schema=schema)
        cmds.append(command)
    if transfer_interval:
        command = 'snmp-server mib bulkstat transfer-id {name} transfer_interval {transfer_interval}'.format(name=name, transfer_interval=transfer_interval)
        cmds.append(command)
    if not any([buffer_size, enable, format_schemaASCI, retry, retain, schema, transfer_interval]) and name:
        command = 'snmp-server mib bulkstat transfer-id {name}'.format(name=name)
        cmds.append(command)
    return cmds