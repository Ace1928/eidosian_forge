from __future__ import absolute_import, division, print_function
import codecs
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_owning_resource, rest_vserver
def get_next_records(self, api):
    """
            Gather next set of ONTAP information for the specified api
            Input for REST APIs call : (api, data)
            return gather_subset_info
        """
    data = {}
    gather_subset_info, error = self.rest_api.get(api, data)
    if error:
        self.module.fail_json(msg=error)
    return gather_subset_info