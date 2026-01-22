from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def monitoring_npu_hpe(data, fos):
    vdom = data['vdom']
    monitoring_npu_hpe_data = data['monitoring_npu_hpe']
    monitoring_npu_hpe_data = flatten_multilists_attributes(monitoring_npu_hpe_data)
    filtered_data = underscore_to_hyphen(filter_monitoring_npu_hpe_data(monitoring_npu_hpe_data))
    return fos.set('monitoring', 'npu-hpe', data=filtered_data, vdom=vdom)