from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def wanopt_remote_storage(data, fos):
    vdom = data['vdom']
    wanopt_remote_storage_data = data['wanopt_remote_storage']
    filtered_data = underscore_to_hyphen(filter_wanopt_remote_storage_data(wanopt_remote_storage_data))
    return fos.set('wanopt', 'remote-storage', data=filtered_data, vdom=vdom)