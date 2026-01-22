from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_authentication_setting_data(json):
    option_list = ['active_auth_scheme', 'auth_https', 'captive_portal', 'captive_portal_ip', 'captive_portal_ip6', 'captive_portal_port', 'captive_portal_ssl_port', 'captive_portal_type', 'captive_portal6', 'cert_auth', 'cert_captive_portal', 'cert_captive_portal_ip', 'cert_captive_portal_port', 'cookie_max_age', 'cookie_refresh_div', 'dev_range', 'ip_auth_cookie', 'persistent_cookie', 'sso_auth_scheme', 'update_time', 'user_cert_ca']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary