from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.acls.acls import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def collect_remarks(aces):
    """makes remarks list per ace"""
    ace_entry = []
    ace_rem = []
    rem = {}
    for i in aces:
        if i.get('is_remark_for'):
            if not rem.get(i.get('is_remark_for')):
                rem[i.get('is_remark_for')] = {'remarks': []}
                rem[i.get('is_remark_for')]['remarks'].append(i.get('the_remark'))
            else:
                rem[i.get('is_remark_for')]['remarks'].append(i.get('the_remark'))
        else:
            if rem:
                if rem.get(i.get('sequence')):
                    ace_rem = rem.pop(i.get('sequence'))
                    i['remarks'] = ace_rem.get('remarks')
            ace_entry.append(i)
    if rem:
        pending_rem = rem.get('remark')
        ace_entry.append({'remarks': pending_rem.get('remarks')})
    return ace_entry