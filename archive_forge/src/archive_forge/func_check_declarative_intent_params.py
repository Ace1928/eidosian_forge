from __future__ import absolute_import, division, print_function
import re
import time
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.validation import check_required_one_of
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def check_declarative_intent_params(want, module, result):
    have = None
    obj_interface = list()
    is_delay = False
    for w in want:
        if w.get('associated_interfaces') is None:
            continue
        if result['changed'] and (not is_delay):
            time.sleep(module.params['delay'])
            is_delay = True
        if have is None:
            have = map_config_to_obj(module)
        obj_in_have = search_obj_in_list(w['vlan_id'], have)
        if obj_in_have:
            for obj in obj_in_have:
                obj_interface.extend(obj['interfaces'])
    for w in want:
        if w.get('associated_interfaces') is None:
            continue
        for i in w['associated_interfaces']:
            if set(obj_interface) - set(w['associated_interfaces']) != set([]):
                module.fail_json(msg='Interface {0} not configured on vlan {1}'.format(i, w['vlan_id']))