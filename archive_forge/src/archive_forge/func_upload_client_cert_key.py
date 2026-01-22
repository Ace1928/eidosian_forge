from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def upload_client_cert_key(self, client_cert, client_key):
    client_cert_file = open(client_cert)
    private_key_file = open(client_key)
    client_cert_content = client_cert_file.read()
    private_key_content = private_key_file.read()
    client_cert_file.close()
    private_key_file.close()
    try:
        self.crypto_mgr.UploadClientCert(cluster=self.key_provider_id, certificate=client_cert_content, privateKey=private_key_content)
    except Exception as e:
        self.module.fail_json(msg="Failed to upload client certificate and private key for key provider '%s' with exception: %s" % (self.key_provider_id.id, to_native(e)))