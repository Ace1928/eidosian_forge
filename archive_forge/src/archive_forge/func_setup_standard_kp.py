from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def setup_standard_kp(self, kp_name, kms_info_list, proxy_user_config_dict):
    kp_id = self.create_key_provider_id(kp_name)
    for kms_info in kms_info_list:
        server_cert = None
        kms_server = self.create_kmip_server_info(kms_info, proxy_user_config_dict)
        kms_spec = self.create_kmip_server_spec(kp_id, kms_server, proxy_user_config_dict.get('kms_password'))
        try:
            self.crypto_mgr.RegisterKmipServer(server=kms_spec)
        except Exception as e:
            self.module.fail_json(msg="Failed to add Standard Key Provider '%s' with exception: %s" % (kp_name, to_native(e)))
        try:
            server_cert = self.crypto_mgr.RetrieveKmipServerCert(keyProvider=kp_id, server=kms_server).certificate
        except Exception as e:
            self.module.fail_json(msg='Failed to retrieve KMS server certificate with exception: %s' % to_native(e))
        if not server_cert:
            self.module.fail_json(msg="Got empty KMS server certificate: '%s'" % server_cert)
        try:
            self.crypto_mgr.UploadKmipServerCert(cluster=kp_id, certificate=server_cert)
        except Exception as e:
            self.module.fail_json(msg="Failed to upload KMS server certificate for key provider '%s' with exception: %s" % (kp_name, to_native(e)))
    return kp_id