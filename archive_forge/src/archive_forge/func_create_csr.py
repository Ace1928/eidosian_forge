from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def create_csr(module, array):
    """Construct a Certificate Signing Request

    Output the result to a specified file
    """
    changed = True
    current_attr = list(array.get_certificates(names=[module.params['name']]).items)[0]
    try:
        if module.params['common_name'] and module.params['common_name'] != getattr(current_attr, 'common_name', None):
            current_attr.common_name = module.params['common_name']
    except AttributeError:
        pass
    try:
        if module.params['country'] and module.params['country'] != getattr(current_attr, 'country', None):
            current_attr.country = module.params['country']
    except AttributeError:
        pass
    try:
        if module.params['email'] and module.params['email'] != getattr(current_attr, 'email', None):
            current_attr.email = module.params['email']
    except AttributeError:
        pass
    try:
        if module.params['locality'] and module.params['locality'] != getattr(current_attr, 'locality', None):
            current_attr.locality = module.params['locality']
    except AttributeError:
        pass
    try:
        if module.params['province'] and module.params['province'] != getattr(current_attr, 'state', None):
            current_attr.state = module.params['province']
    except AttributeError:
        pass
    try:
        if module.params['organization'] and module.params['organization'] != getattr(current_attr, 'organization', None):
            current_attr.organization = module.params['organization']
    except AttributeError:
        pass
    try:
        if module.params['org_unit'] and module.params['org_unit'] != getattr(current_attr, 'organizational_unit', None):
            current_attr.organizational_unit = module.params['org_unit']
    except AttributeError:
        pass
    if not module.check_mode:
        certificate = flasharray.CertificateSigningRequestPost(certificate={'name': module.params['name']}, common_name=getattr(current_attr, 'common_name', None), country=getattr(current_attr, 'country', None), email=getattr(current_attr, 'email', None), locality=getattr(current_attr, 'locality', None), state=getattr(current_attr, 'state', None), organization=getattr(current_attr, 'organization', None), organizational_unit=getattr(current_attr, 'organizational_unit', None))
        csr = list(array.post_certificates_certificate_signing_requests(certificate=certificate).items)[0].certificate_signing_request
        csr_file = open(module.params['export_file'], 'w')
        csr_file.write(csr)
        csr_file.close()
    module.exit_json(changed=changed)