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
def filter_webfilter_profile_data(json):
    option_list = ['antiphish', 'comment', 'extended_log', 'feature_set', 'file_filter', 'ftgd_wf', 'https_replacemsg', 'inspection_mode', 'log_all_url', 'name', 'options', 'override', 'ovrd_perm', 'post_action', 'replacemsg_group', 'url_extraction', 'web', 'web_antiphishing_log', 'web_content_log', 'web_extended_all_action_log', 'web_filter_activex_log', 'web_filter_applet_log', 'web_filter_command_block_log', 'web_filter_cookie_log', 'web_filter_cookie_removal_log', 'web_filter_js_log', 'web_filter_jscript_log', 'web_filter_referer_log', 'web_filter_unknown_log', 'web_filter_vbs_log', 'web_ftgd_err_log', 'web_ftgd_quota_usage', 'web_invalid_domain_log', 'web_url_log', 'wisp', 'wisp_algorithm', 'wisp_servers', 'youtube_channel_filter', 'youtube_channel_status']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary