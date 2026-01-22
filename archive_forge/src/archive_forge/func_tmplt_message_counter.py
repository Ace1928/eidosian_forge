from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_message_counter(verb):
    cmd = 'logging message-counter'
    if verb.get('message_counter'):
        cmd += ' {message_counter}'.format(message_counter=verb['message_counter'])
    return cmd