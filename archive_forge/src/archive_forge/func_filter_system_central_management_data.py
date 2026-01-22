from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_system_central_management_data(json):
    option_list = ['allow_monitor', 'allow_push_configuration', 'allow_push_firmware', 'allow_remote_firmware_upgrade', 'ca_cert', 'enc_algorithm', 'fmg', 'fmg_source_ip', 'fmg_source_ip6', 'fmg_update_port', 'fortigate_cloud_sso_default_profile', 'include_default_servers', 'interface', 'interface_select_method', 'local_cert', 'mode', 'schedule_config_restore', 'schedule_script_restore', 'serial_number', 'server_list', 'type', 'vdom']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary