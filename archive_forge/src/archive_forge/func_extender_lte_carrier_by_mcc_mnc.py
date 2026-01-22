from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def extender_lte_carrier_by_mcc_mnc(data, fos):
    vdom = data['vdom']
    extender_lte_carrier_by_mcc_mnc_data = data['extender_lte_carrier_by_mcc_mnc']
    filtered_data = underscore_to_hyphen(filter_extender_lte_carrier_by_mcc_mnc_data(extender_lte_carrier_by_mcc_mnc_data))
    converted_data = valid_attr_to_invalid_attrs(filtered_data)
    return fos.set('extender', 'lte-carrier-by-mcc-mnc', data=converted_data, vdom=vdom)