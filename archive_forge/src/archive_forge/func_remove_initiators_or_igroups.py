from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def remove_initiators_or_igroups(self, uuid, option, current_names, mapping):
    """
        Removes current names from igroup unless they are still desired
        :return: None
        """
    self.check_option_is_valid(option)
    for name in current_names:
        if name not in self.parameters.get(option, list()):
            if self.use_rest:
                self.delete_initiator_or_igroup_rest(uuid, option, mapping[name])
            else:
                self.modify_initiator(name, 'igroup-remove')