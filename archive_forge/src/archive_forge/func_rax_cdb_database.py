from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import rax_argument_spec, rax_required_together, rax_to_dict, setup_rax_module
def rax_cdb_database(module, state, cdb_id, name, character_set, collate):
    if state == 'present':
        save_database(module, cdb_id, name, character_set, collate)
    elif state == 'absent':
        delete_database(module, cdb_id, name)