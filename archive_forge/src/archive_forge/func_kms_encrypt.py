from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import GcpSession
def kms_encrypt(self, module):
    payload = {'plaintext': module.params['plaintext']}
    if module.params['additional_authenticated_data']:
        payload['additionalAuthenticatedData'] = module.params['additional_authenticated_data']
    auth = GcpSession(module, 'cloudkms')
    url = 'https://cloudkms.googleapis.com/v1/projects/{projects}/locations/{locations}/keyRings/{key_ring}/cryptoKeys/{crypto_key}:encrypt'.format(**module.params)
    response = auth.post(url, body=payload)
    return response.json()['ciphertext']