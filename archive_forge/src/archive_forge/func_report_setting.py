from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def report_setting(data, fos):
    vdom = data['vdom']
    report_setting_data = data['report_setting']
    report_setting_data = flatten_multilists_attributes(report_setting_data)
    filtered_data = underscore_to_hyphen(filter_report_setting_data(report_setting_data))
    return fos.set('report', 'setting', data=filtered_data, vdom=vdom)