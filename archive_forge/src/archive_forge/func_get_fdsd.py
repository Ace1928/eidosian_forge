from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_fdsd(self):
    """
        Get File Directory Security Descriptor
        """
    api = 'private/cli/vserver/security/file-directory/ntfs'
    query = {'ntfs-sd': self.parameters['name'], 'vserver': self.parameters['vserver']}
    message, error = self.rest_api.get(api, query)
    records, error = rrh.check_for_0_or_more_records(api, message, error)
    if error:
        self.module.fail_json(msg=error)
    return records if records else None