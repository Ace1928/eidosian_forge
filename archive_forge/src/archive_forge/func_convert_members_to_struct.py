from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def convert_members_to_struct(member_spec):
    """ Transforms the members list of the Network module arguments into a
    valid WAPI struct. This function will change arguments into the valid
    wapi structure of the format:
        {
            network: 10.1.1.0/24
            members:
                [
                    {'_struct': 'dhcpmember', 'name': 'member_name1'},
                    {'_struct': 'dhcpmember', 'name': 'member_name2'}
                    {'_struct': 'dhcpmember', 'name': '...'}
                ]
        }
    """
    if 'members' in member_spec.keys():
        member_spec['members'] = [{'_struct': 'dhcpmember', 'name': k['name']} for k in member_spec['members']]
    return member_spec