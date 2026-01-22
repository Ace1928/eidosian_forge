from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_zapi_key_and_value(key, value):
    zapi_key = self.na_helper.zapi_string_keys.get(key)
    if zapi_key is not None:
        return (zapi_key, value)
    zapi_key = self.na_helper.zapi_bool_keys.get(key)
    if zapi_key is not None:
        return (zapi_key, self.na_helper.get_value_for_bool(from_zapi=False, value=value))
    zapi_key = self.na_helper.zapi_int_keys.get(key)
    if zapi_key is not None:
        return (zapi_key, self.na_helper.get_value_for_int(from_zapi=False, value=value))
    raise KeyError(key)