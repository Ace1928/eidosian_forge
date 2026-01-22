from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_antivirus_profile_data(json):
    option_list = ['analytics_accept_filetype', 'analytics_bl_filetype', 'analytics_db', 'analytics_ignore_filetype', 'analytics_max_upload', 'analytics_wl_filetype', 'av_block_log', 'av_virus_log', 'cifs', 'comment', 'content_disarm', 'ems_threat_feed', 'extended_log', 'external_blocklist', 'external_blocklist_archive_scan', 'external_blocklist_enable_all', 'feature_set', 'fortiai_error_action', 'fortiai_timeout_action', 'fortindr_error_action', 'fortindr_timeout_action', 'fortisandbox_error_action', 'fortisandbox_max_upload', 'fortisandbox_mode', 'fortisandbox_timeout_action', 'ftgd_analytics', 'ftp', 'http', 'imap', 'inspection_mode', 'mapi', 'mobile_malware_db', 'nac_quar', 'name', 'nntp', 'outbreak_prevention', 'outbreak_prevention_archive_scan', 'pop3', 'replacemsg_group', 'scan_mode', 'smb', 'smtp', 'ssh']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary