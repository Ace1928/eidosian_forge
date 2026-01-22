from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
def filter_icap_profile_data(json):
    option_list = ['response_204', 'size_limit_204', 'chunk_encap', 'comment', 'extension_feature', 'file_transfer', 'file_transfer_failure', 'file_transfer_path', 'file_transfer_server', 'icap_block_log', 'icap_headers', 'methods', 'name', 'preview', 'preview_data_length', 'replacemsg_group', 'request', 'request_failure', 'request_path', 'request_server', 'respmod_default_action', 'respmod_forward_rules', 'response', 'response_failure', 'response_path', 'response_req_hdr', 'response_server', 'scan_progress_interval', 'streaming_content_bypass', 'timeout']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary