from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_cert(module, blade):
    """Create certificate"""
    changed = True
    if not module.check_mode:
        try:
            body = CertificatePost(certificate=module.params['contents'], certificate_type='external')
            blade.certificates.create_certificates(names=[module.params['name']], certificate=body)
        except Exception:
            module.fail_json(msg='Failed to create certificate {0}.'.format(module.params['name']))
    module.exit_json(changed=changed)