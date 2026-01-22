from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def eradicate_bucket(module, blade):
    """Eradicate Bucket"""
    changed = True
    if not module.check_mode:
        try:
            blade.buckets.delete_buckets(names=[module.params['name']])
        except Exception:
            module.fail_json(msg='Object Store Bucket {0}: Eradication failed'.format(module.params['name']))
    module.exit_json(changed=changed)