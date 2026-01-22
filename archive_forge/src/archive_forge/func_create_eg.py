from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.vexata import (
def create_eg(module, array):
    """"Create a new export group."""
    changed = False
    eg_name = module.params['name']
    vg_id = get_vg_id(module, array)
    ig_id = get_ig_id(module, array)
    pg_id = get_pg_id(module, array)
    if module.check_mode:
        module.exit_json(changed=changed)
    try:
        eg = array.create_eg(eg_name, 'Ansible export group', (vg_id, ig_id, pg_id))
        if eg:
            module.log(msg='Created export group {0}'.format(eg_name))
            changed = True
        else:
            raise Exception
    except Exception:
        module.fail_json(msg='Export group {0} create failed.'.format(eg_name))
    module.exit_json(changed=changed)