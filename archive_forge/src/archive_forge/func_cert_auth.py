from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def cert_auth(self, path=None, payload='', method=None):
    """Perform APIC signature-based authentication, not the expected SSL client certificate authentication."""
    if method is None:
        method = self.params.get('method').upper()
    if path is None:
        path = self.path
    path = '/' + path.lstrip('/')
    if payload is None:
        payload = ''
    try:
        if HAS_CRYPTOGRAPHY:
            key = self.params.get('private_key').encode()
            sig_key = serialization.load_pem_private_key(key, password=None, backend=default_backend())
        else:
            sig_key = load_privatekey(FILETYPE_PEM, self.params.get('private_key'))
    except Exception:
        if os.path.exists(self.params.get('private_key')):
            try:
                permission = 'r'
                if HAS_CRYPTOGRAPHY:
                    permission = 'rb'
                with open(self.params.get('private_key'), permission) as fh:
                    private_key_content = fh.read()
            except Exception:
                self.module.fail_json(msg="Cannot open private key file '{private_key}'.".format_map(self.params))
            try:
                if HAS_CRYPTOGRAPHY:
                    sig_key = serialization.load_pem_private_key(private_key_content, password=None, backend=default_backend())
                else:
                    sig_key = load_privatekey(FILETYPE_PEM, private_key_content)
            except Exception:
                self.module.fail_json(msg="Cannot load private key file '{private_key}'.".format_map(self.params))
            if self.params.get('certificate_name') is None:
                self.params['certificate_name'] = os.path.basename(os.path.splitext(self.params.get('private_key'))[0])
        else:
            self.module.fail_json(msg='Provided private key {private_key} does not appear to be a private key or provided file does not exist.'.format_map(self.params))
    if self.params.get('certificate_name') is None:
        self.params['certificate_name'] = 'admin' if self.params.get('username') is None else self.params.get('username')
    sig_request = method + path + payload
    if HAS_CRYPTOGRAPHY:
        sig_signature = sig_key.sign(sig_request.encode(), padding.PKCS1v15(), hashes.SHA256())
    else:
        sig_signature = sign(sig_key, sig_request, 'sha256')
    sig_dn = 'uni/userext/user-{username}/usercert-{certificate_name}'.format_map(self.params)
    self.headers['Cookie'] = 'APIC-Certificate-Algorithm=v1.0; ' + 'APIC-Certificate-DN={0}; '.format(sig_dn) + 'APIC-Certificate-Fingerprint=fingerprint; ' + 'APIC-Request-Signature={0}'.format(to_native(base64.b64encode(sig_signature)))