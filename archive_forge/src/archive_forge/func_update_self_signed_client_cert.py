from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def update_self_signed_client_cert(self, dest_path):
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            self.module.fail_json(msg="Specified destination path '%s' not exist, but failed to create it with exception: %s" % (dest_path, to_native(e)))
    client_cert_file_path = os.path.join(dest_path, self.key_provider_id.id + '_self_signed_cert.pem')
    client_cert = self.crypto_mgr.RetrieveSelfSignedClientCert(self.key_provider_id)
    if not client_cert:
        try:
            client_cert = self.crypto_mgr.GenerateSelfSignedClientCert(self.key_provider_id)
        except Exception as e:
            self.module.fail_json(msg='Generate self signed client certificate failed with exception: %s' % to_native(e))
    if not client_cert:
        self.module.fail_json(msg="Generated self signed client certificate is empty '%s'" % client_cert)
    try:
        self.crypto_mgr.UpdateSelfSignedClientCert(self.key_provider_id, client_cert)
    except Exception as e:
        self.module.fail_json(msg='Update self signed client cert failed with exception: %s' % to_native(e))
    client_cert_file = open(client_cert_file_path, 'w')
    client_cert_file.write(client_cert)
    client_cert_file.close()
    return client_cert_file_path