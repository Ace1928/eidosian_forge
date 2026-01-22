from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def web_proxy_global(data, fos):
    vdom = data['vdom']
    web_proxy_global_data = data['web_proxy_global']
    web_proxy_global_data = flatten_multilists_attributes(web_proxy_global_data)
    filtered_data = underscore_to_hyphen(filter_web_proxy_global_data(web_proxy_global_data))
    return fos.set('web-proxy', 'global', data=filtered_data, vdom=vdom)