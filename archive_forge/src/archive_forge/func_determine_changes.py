from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def determine_changes(self):
    """Determine certificates that need to be added or removed from storage system's server certificates database."""
    if not self.is_proxy():
        self.check_controller()
    existing_certificates = self.get_current_certificates()
    expected = self.get_expected_certificates()
    certificates = expected['certificates']
    changes = {'change_required': False, 'signed_cert': True if certificates else False, 'private_key': expected['private_key'], 'public_cert': None, 'add_certs': [], 'remove_certs': []}
    if certificates:
        for existing_certificate_subject, existing_certificate in existing_certificates.items():
            changes['remove_certs'].append(existing_certificate['alias'])
        last_certificate_index = len(certificates) - 1
        for certificate_index, certificate in enumerate(certificates):
            for existing_certificate_subject, existing_certificate in existing_certificates.items():
                if certificate_index == last_certificate_index:
                    if existing_certificate['alias'] == 'jetty':
                        if certificate['fingerprint'] != existing_certificate['shaFingerprint'] and certificate['fingerprint'] != existing_certificate['sha256Fingerprint']:
                            changes['change_required'] = True
                        changes['public_cert'] = certificate
                        changes['remove_certs'].remove(existing_certificate['alias'])
                        break
                elif certificate['alias'] == existing_certificate['alias']:
                    if certificate['fingerprint'] != existing_certificate['shaFingerprint'] and certificate['fingerprint'] != existing_certificate['sha256Fingerprint']:
                        changes['add_certs'].append(certificate)
                        changes['change_required'] = True
                    changes['remove_certs'].remove(existing_certificate['alias'])
                    break
            else:
                changes['add_certs'].append(certificate)
                changes['change_required'] = True
    elif self.is_public_server_certificate_signed():
        changes['change_required'] = True
    return changes