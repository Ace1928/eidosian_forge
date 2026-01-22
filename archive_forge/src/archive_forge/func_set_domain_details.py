from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import (
def set_domain_details(self, domain_details):
    if domain_details.get('verificationMethod'):
        self.verification_method = domain_details['verificationMethod'].lower()
    self.domain_status = domain_details['verificationStatus']
    self.ov_eligible = domain_details.get('ovEligible')
    self.ov_days_remaining = calculate_days_remaining(domain_details.get('ovExpiry'))
    self.ev_eligible = domain_details.get('evEligible')
    self.ev_days_remaining = calculate_days_remaining(domain_details.get('evExpiry'))
    self.client_id = domain_details['clientId']
    if self.verification_method == 'dns' and domain_details.get('dnsMethod'):
        self.dns_location = domain_details['dnsMethod']['recordDomain']
        self.dns_resource_type = domain_details['dnsMethod']['recordType']
        self.dns_contents = domain_details['dnsMethod']['recordValue']
    elif self.verification_method == 'web_server' and domain_details.get('webServerMethod'):
        self.file_location = domain_details['webServerMethod']['fileLocation']
        self.file_contents = domain_details['webServerMethod']['fileContents']
    elif self.verification_method == 'email' and domain_details.get('emailMethod'):
        self.emails = domain_details['emailMethod']