from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def initiator_to_dict(self, initiator_obj):
    """
        converts initiator class object to dict
        :return: reconstructed initiator dict
        """
    known_params = ['initiator_name', 'alias', 'initiator_id', 'volume_access_groups', 'volume_access_group_id', 'attributes']
    initiator_dict = {}
    for param in known_params:
        initiator_dict[param] = getattr(initiator_obj, param, None)
    if initiator_dict['volume_access_groups'] is not None:
        if len(initiator_dict['volume_access_groups']) == 1:
            initiator_dict['volume_access_group_id'] = initiator_dict['volume_access_groups'][0]
        elif len(initiator_dict['volume_access_groups']) > 1:
            self.module.fail_json(msg='Only 1 access group is supported, found: %s' % repr(initiator_obj))
    del initiator_dict['volume_access_groups']
    return initiator_dict