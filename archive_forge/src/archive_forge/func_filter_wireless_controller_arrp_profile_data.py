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
def filter_wireless_controller_arrp_profile_data(json):
    option_list = ['comment', 'darrp_optimize', 'darrp_optimize_schedules', 'include_dfs_channel', 'include_weather_channel', 'monitor_period', 'name', 'override_darrp_optimize', 'selection_period', 'threshold_ap', 'threshold_channel_load', 'threshold_noise_floor', 'threshold_rx_errors', 'threshold_spectral_rssi', 'threshold_tx_retries', 'weight_channel_load', 'weight_dfs_channel', 'weight_managed_ap', 'weight_noise_floor', 'weight_rogue_ap', 'weight_spectral_rssi', 'weight_weather_channel']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary