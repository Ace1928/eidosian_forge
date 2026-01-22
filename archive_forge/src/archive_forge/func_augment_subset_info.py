from __future__ import absolute_import, division, print_function
import codecs
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_owning_resource, rest_vserver
def augment_subset_info(self, subset, subset_info):
    if subset == 'private/cli/vserver/security/file-directory':
        subset_info = self.strip_dacls(subset_info)
    if subset == 'storage/luns':
        self.add_naa_id(subset_info)
    return subset_info