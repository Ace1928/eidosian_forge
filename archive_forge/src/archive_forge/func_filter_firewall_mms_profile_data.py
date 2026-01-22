from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_firewall_mms_profile_data(json):
    option_list = ['avnotificationtable', 'bwordtable', 'carrier_endpoint_prefix', 'carrier_endpoint_prefix_range_max', 'carrier_endpoint_prefix_range_min', 'carrier_endpoint_prefix_string', 'carrierendpointbwltable', 'comment', 'dupe', 'extended_utm_log', 'flood', 'mm1', 'mm1_addr_hdr', 'mm1_addr_source', 'mm1_convert_hex', 'mm1_outbreak_prevention', 'mm1_retr_dupe', 'mm1_retrieve_scan', 'mm1comfortamount', 'mm1comfortinterval', 'mm1oversizelimit', 'mm3', 'mm3_outbreak_prevention', 'mm3oversizelimit', 'mm4', 'mm4_outbreak_prevention', 'mm4oversizelimit', 'mm7', 'mm7_addr_hdr', 'mm7_addr_source', 'mm7_convert_hex', 'mm7_outbreak_prevention', 'mm7comfortamount', 'mm7comfortinterval', 'mm7oversizelimit', 'mms_antispam_mass_log', 'mms_av_block_log', 'mms_av_oversize_log', 'mms_av_virus_log', 'mms_carrier_endpoint_filter_log', 'mms_checksum_log', 'mms_checksum_table', 'mms_notification_log', 'mms_web_content_log', 'mmsbwordthreshold', 'name', 'notif_msisdn', 'notification', 'outbreak_prevention', 'remove_blocked_const_length', 'replacemsg_group']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary