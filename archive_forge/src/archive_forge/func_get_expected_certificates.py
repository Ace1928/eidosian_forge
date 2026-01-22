from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def get_expected_certificates(self):
    """Determine effected certificates and return certificate list in the required submission order."""
    certificates_info = {}
    existing_certificates = self.get_current_certificates()
    private_key = None
    if self.certificates:
        for path in self.certificates:
            info = self.certificate_info_from_file(path)
            if 'private_key' in info.keys():
                if private_key is not None and info['private_key'] != private_key:
                    self.module.fail_json(msg='Multiple private keys have been provided! Array [%s]' % self.ssid)
                else:
                    private_key = info.pop('private_key')
            certificates_info.update(info)
    ordered_certificates_info = [dict] * len(certificates_info)
    ordered_certificates_info_index = len(certificates_info) - 1
    while certificates_info:
        for certificate_subject in certificates_info.keys():
            remaining_issuer_list = [info['issuer'] for subject, info in existing_certificates.items()]
            for subject, info in certificates_info.items():
                remaining_issuer_list.append(info['issuer'])
            if certificate_subject not in remaining_issuer_list:
                ordered_certificates_info[ordered_certificates_info_index] = certificates_info[certificate_subject]
                certificates_info.pop(certificate_subject)
                ordered_certificates_info_index -= 1
                break
        else:
            for certificate_subject in certificates_info.keys():
                ordered_certificates_info[ordered_certificates_info_index] = certificates_info[certificate_subject]
                ordered_certificates_info_index -= 1
            break
    return {'private_key': private_key, 'certificates': ordered_certificates_info}