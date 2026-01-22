from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, rax_to_dict, setup_rax_module
def rax_cdb_user(module, state, cdb_id, name, password, databases, host):
    if state == 'present':
        save_user(module, cdb_id, name, password, databases, host)
    elif state == 'absent':
        delete_user(module, cdb_id, name)