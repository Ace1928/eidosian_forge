from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_certgrp(module, blade):
    """Create certifcate group"""
    changed = True
    if not module.check_mode:
        try:
            blade.certificate_groups.create_certificate_groups(names=[module.params['name']])
        except Exception:
            module.fail_json(msg='Failed to create certificate group {0}.'.format(module.params['name']))
        if module.params['certificates']:
            try:
                blade.certificate_groups.add_certificate_group_certificates(certificate_names=module.params['certificates'], certificate_group_names=[module.params['name']])
            except Exception:
                blade.certificate_groups.delete_certificate_groups(names=[module.params['name']])
                module.fail_json(msg='Failed to add certifcates {0}. Please check they all exist'.format(module.params['certificates']))
    module.exit_json(changed=changed)