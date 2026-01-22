from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def associate_host_map(self, types):
    """
        Check if there are hosts or paths to be associated with the subsystem
        """
    action_add_dict = {}
    action_remove_dict = {}
    for type in types:
        current = None
        if self.parameters.get(type):
            if self.use_rest:
                if self.subsystem_uuid:
                    current = self.get_subsystem_host_map_rest(type)
            else:
                current = self.get_subsystem_host_map(type)
            if current:
                add_items = self.na_helper.get_modified_attributes(current, self.parameters, get_list_diff=True).get(type)
                remove_items = [item for item in current[type] if item not in self.parameters.get(type)]
            else:
                add_items = self.parameters[type]
                remove_items = {}
            if add_items:
                action_add_dict[type] = add_items
                self.na_helper.changed = True
            if remove_items:
                action_remove_dict[type] = remove_items
                self.na_helper.changed = True
    return (action_add_dict, action_remove_dict)