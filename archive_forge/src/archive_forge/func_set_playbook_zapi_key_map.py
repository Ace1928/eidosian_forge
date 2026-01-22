from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def set_playbook_zapi_key_map(self):
    self.na_helper.zapi_string_keys = {'node_name': 'node-name', 'transport': 'transport', 'post_url': 'post-url', 'from_address': 'from', 'proxy_url': 'proxy-url'}
    self.na_helper.zapi_int_keys = {'retry_count': 'retry-count', 'max_http_size': 'max-http-size', 'max_smtp_size': 'max-smtp-size'}
    self.na_helper.zapi_list_keys = {'noteto': ('noteto', 'mail-address'), 'mail_hosts': ('mail-hosts', 'string'), 'partner_addresses': ('partner-address', 'mail-address'), 'to_addresses': ('to', 'mail-address')}
    self.na_helper.zapi_bool_keys = {'support': 'is-support-enabled', 'hostname_in_subject': 'is-node-in-subject', 'nht_data_enabled': 'is-nht-data-enabled', 'perf_data_enabled': 'is-perf-data-enabled', 'reminder_enabled': 'is-reminder-enabled', 'private_data_removed': 'is-private-data-removed', 'local_collection_enabled': 'is-local-collection-enabled', 'ondemand_enabled': 'is-ondemand-enabled', 'validate_digital_certificate': 'validate-digital-certificate'}