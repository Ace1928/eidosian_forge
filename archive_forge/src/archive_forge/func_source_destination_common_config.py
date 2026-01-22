from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def source_destination_common_config(config_data, command, attr):
    if config_data[attr].get('address'):
        command += ' {address}'.format(**config_data[attr])
        if config_data[attr].get('wildcard_bits'):
            command += ' {wildcard_bits}'.format(**config_data[attr])
    elif config_data[attr].get('any'):
        command += ' any'.format(**config_data[attr])
    elif config_data[attr].get('host'):
        command += ' host {host}'.format(**config_data[attr])
    elif config_data[attr].get('object_group'):
        command += ' object-group {object_group}'.format(**config_data[attr])
    if config_data[attr].get('port_protocol'):
        if config_data[attr].get('port_protocol').get('range'):
            command += ' range {0} {1}'.format(config_data[attr]['port_protocol']['range'].get('start'), config_data[attr]['port_protocol']['range'].get('end'))
        else:
            port_proto_type = list(config_data[attr]['port_protocol'].keys())[0]
            command += ' {0} {1}'.format(port_proto_type, config_data[attr]['port_protocol'][port_proto_type])
    return command