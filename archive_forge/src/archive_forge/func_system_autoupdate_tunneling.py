from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_autoupdate_tunneling(data, fos):
    vdom = data['vdom']
    system_autoupdate_tunneling_data = data['system_autoupdate_tunneling']
    filtered_data = underscore_to_hyphen(filter_system_autoupdate_tunneling_data(system_autoupdate_tunneling_data))
    return fos.set('system.autoupdate', 'tunneling', data=filtered_data, vdom=vdom)