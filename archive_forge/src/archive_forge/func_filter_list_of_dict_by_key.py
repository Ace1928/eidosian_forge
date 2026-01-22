from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def filter_list_of_dict_by_key(self, records, key, value):
    matched = list()
    for record in records:
        if key in record and record[key] == value:
            matched.append(record)
        if key not in record and self.parameters['fail_on_key_not_found']:
            msg = 'Error: key %s not found in %s' % (key, repr(record))
            self.module.fail_json(msg=msg)
    return matched