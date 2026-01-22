from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def rem_tmplt_tls_servers(config_data):
    command = 'logging tls-server'
    if config_data.get('name'):
        command += ' {name}'.format(name=config_data['name'])
    return command