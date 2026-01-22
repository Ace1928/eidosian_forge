from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (rax_argument_spec, rax_required_together, rax_to_dict,
def cloud_identity(module, state, identity):
    instance = dict(authenticated=identity.authenticated, credentials=identity._creds_file)
    changed = False
    instance.update(rax_to_dict(identity))
    instance['services'] = instance.get('services', {}).keys()
    if state == 'present':
        if not identity.authenticated:
            module.fail_json(msg='Credentials could not be verified!')
    module.exit_json(changed=changed, identity=instance)