from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def get_and_filter_info(self, name):
    """
        Get data
        If filter is present, only return the records that are matched
        return output as json
        """
    records = self.get_info(name)
    if self.parameters.get('filter') is None:
        return records
    matched = self.filter_records(records, self.parameters.get('filter'))
    return matched