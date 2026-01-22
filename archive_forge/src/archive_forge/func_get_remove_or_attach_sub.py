from __future__ import absolute_import, division, print_function
import json
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text, to_native
def get_remove_or_attach_sub(module, manifest):
    changed = False
    subs = get_subs(module, manifest)
    if subs:
        if module.params['pool_state'] == 'present':
            sub_quantity = sum((s['quantity'] for s in subs))
            while sub_quantity > module.params['quantity']:
                if not module.check_mode:
                    remove_sub(module, manifest, subs[0])
                else:
                    changed = True
                    break
                changed = True
                subs = get_subs(module, manifest)
                sub_quantity = sum((s['quantity'] for s in subs))
            if sub_quantity < module.params['quantity']:
                difference = module.params['quantity'] - sub_quantity
                if not module.check_mode:
                    attach_sub(module, manifest, difference)
                changed = True
        elif module.params['pool_state'] == 'absent':
            if not module.check_mode:
                for sub in subs:
                    remove_sub(module, manifest, sub)
            changed = True
    elif module.params['pool_state'] == 'present':
        if not module.check_mode:
            attach_sub(module, manifest, module.params['quantity'])
        changed = True
    return changed