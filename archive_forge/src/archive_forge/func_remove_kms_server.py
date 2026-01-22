from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def remove_kms_server(self, key_provider_id, kms_server):
    try:
        self.crypto_mgr.RemoveKmipServer(clusterId=key_provider_id, serverName=kms_server)
    except Exception as e:
        self.module.fail_json(msg="Failed to remove KMIP server '%s' from key provider '%s' with exception: %s" % (kms_server, key_provider_id.id, to_native(e)))