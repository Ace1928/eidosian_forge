from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_correlator_ruleset(config_data):
    command = 'logging correlator ruleset'
    if config_data.get('name'):
        command += '  {name}'.format(name=config_data['name'])
    if config_data.get('rulename'):
        command += '  rulename {rulename}'.format(rulename=config_data['rulename'])
    return command