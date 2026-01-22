from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def system_smc_ntp(data, fos):
    vdom = data['vdom']
    system_smc_ntp_data = data['system_smc_ntp']
    filtered_data = underscore_to_hyphen(filter_system_smc_ntp_data(system_smc_ntp_data))
    return fos.set('system', 'smc-ntp', data=filtered_data, vdom=vdom)