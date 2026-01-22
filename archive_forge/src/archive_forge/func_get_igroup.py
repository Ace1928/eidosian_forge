from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_igroup(self, name):
    """
        Return details about the igroup
        :param:
            name : Name of the igroup

        :return: Details about the igroup. None if not found.
        :rtype: dict
        """
    if self.use_rest:
        return self.get_igroup_rest(name)
    igroup_info = netapp_utils.zapi.NaElement('igroup-get-iter')
    attributes = dict(query={'initiator-group-info': {'initiator-group-name': name, 'vserver': self.parameters['vserver']}})
    igroup_info.translate_struct(attributes)
    current = None
    try:
        result = self.server.invoke_successfully(igroup_info, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching igroup info %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
        igroup_info = result.get_child_by_name('attributes-list')
        initiator_group_info = igroup_info.get_child_by_name('initiator-group-info')
        initiator_names = []
        initiator_objects = []
        if initiator_group_info.get_child_by_name('initiators'):
            current_initiators = initiator_group_info['initiators'].get_children()
            initiator_names = [initiator['initiator-name'] for initiator in current_initiators]
            initiator_objects = [dict(name=initiator['initiator-name'], comment=None) for initiator in current_initiators]
        current = {'initiator_names': initiator_names, 'initiator_objects': initiator_objects, 'name_to_uuid': dict(initiator_names=dict())}
        zapi_to_params = {'vserver': 'vserver', 'initiator-group-os-type': 'os_type', 'initiator-group-portset-name': 'bind_portset', 'initiator-group-type': 'initiator_group_type'}
        for attr in zapi_to_params:
            value = igroup_info.get_child_content(attr)
            if value is not None:
                current[zapi_to_params[attr]] = value
    return current