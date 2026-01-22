from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_volume
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def set_lun_value(self, path, key, value):
    key_to_zapi = dict(comment=('lun-set-comment', 'comment'), qos_policy_group=('lun-set-qos-policy-group', 'qos-policy-group'), qos_adaptive_policy_group=('lun-set-qos-policy-group', 'qos-adaptive-policy-group'), space_allocation=('lun-set-space-alloc', 'enable'), space_reserve=('lun-set-space-reservation-info', 'enable'))
    if key in key_to_zapi:
        zapi, option = key_to_zapi[key]
    else:
        self.module.fail_json(msg='option %s cannot be modified to %s' % (key, value))
    options = dict(path=path)
    if option == 'enable':
        options[option] = self.na_helper.get_value_for_bool(False, value)
    else:
        options[option] = value
    lun_set = netapp_utils.zapi.NaElement.create_node_with_children(zapi, **options)
    try:
        self.server.invoke_successfully(lun_set, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as exc:
        self.module.fail_json(msg='Error setting lun option %s: %s' % (key, to_native(exc)), exception=traceback.format_exc())
    return