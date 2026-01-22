from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def update_kmip(module, array):
    """Update existing KMIP object"""
    changed = False
    current_kmip = list(array.get_kmip(names=[module.params['name']]).items)[0]
    if module.params['certificate'] and current_kmip.certificate.name != module.params['certificate']:
        if array.get_certificates(names=[module.params['certificate']]).status_code != 200:
            module.fail_json(msg='Array certificate {0} does not exist.'.format(module.params['certificate']))
        changed = True
        certificate = module.params['certificate']
    else:
        certificate = current_kmip.certificate.name
    if module.params['uris'] and sorted(current_kmip.uris) != sorted(module.params['uris']):
        changed = True
        uris = sorted(module.params['uris'])
    else:
        uris = sorted(current_kmip.uris)
    if module.params['ca_certificate'] and module.params['ca_certificate'] != current_kmip.ca_certificate:
        changed = True
        ca_cert = module.params['ca_certificate']
    else:
        ca_cert = current_kmip.ca_certificate
    if not module.check_mode:
        if changed:
            kmip = flasharray.KmipPost(uris=uris, ca_certificate=ca_cert, certificate=flasharray.ReferenceNoId(name=certificate))
            res = array.patch_kmip(names=[module.params['name']], kmip=kmip)
            if res.status_code != 200:
                module.fail_json(msg='Updating existing KMIP object {0} failed. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)