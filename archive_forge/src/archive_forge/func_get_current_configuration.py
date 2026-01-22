from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def get_current_configuration(self, update=False):
    """Retrieve the current storage array's global configuration."""
    if self.current_configuration_cache is None or update:
        self.current_configuration_cache = dict()
        try:
            rc, capabilities = self.request('storage-systems/%s/capabilities' % self.ssid)
            self.current_configuration_cache['autoload_capable'] = 'capabilityAutoLoadBalancing' in capabilities['productCapabilities']
            self.current_configuration_cache['cache_block_size_options'] = capabilities['featureParameters']['cacheBlockSizes']
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve storage array capabilities. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
        try:
            rc, host_types = self.request('storage-systems/%s/host-types' % self.ssid)
            self.current_configuration_cache['host_type_options'] = dict()
            for host_type in host_types:
                self.current_configuration_cache['host_type_options'].update({host_type['code'].lower(): host_type['index']})
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve storage array host options. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
        try:
            rc, settings = self.request('storage-systems/%s/graph/xpath-filter?query=/sa' % self.ssid)
            self.current_configuration_cache['cache_settings'] = {'cache_block_size': settings[0]['cache']['cacheBlkSize'], 'cache_flush_threshold': settings[0]['cache']['demandFlushThreshold']}
            self.current_configuration_cache['default_host_type_index'] = settings[0]['defaultHostTypeIndex']
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve cache settings. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
        try:
            rc, array_info = self.request('storage-systems/%s' % self.ssid)
            self.current_configuration_cache['autoload_enabled'] = array_info['autoLoadBalancingEnabled']
            self.current_configuration_cache['host_connectivity_reporting_enabled'] = array_info['hostConnectivityReportingEnabled']
            self.current_configuration_cache['name'] = array_info['name']
        except Exception as error:
            self.module.fail_json(msg='Failed to determine current configuration. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
        try:
            rc, login_banner_message = self.request('storage-systems/%s/login-banner?asFile=false' % self.ssid, ignore_errors=True, json_response=False, headers={'Accept': 'application/octet-stream', 'netapp-client-type': 'Ansible-%s' % ansible_version})
            self.current_configuration_cache['login_banner_message'] = login_banner_message.decode('utf-8').rstrip('\n')
        except Exception as error:
            self.module.fail_json(msg='Failed to determine current login banner message. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
        try:
            rc, hardware_inventory = self.request('storage-systems/%s/hardware-inventory' % self.ssid)
            self.current_configuration_cache['controller_shelf_reference'] = hardware_inventory['trays'][0]['trayRef']
            self.current_configuration_cache['controller_shelf_id'] = hardware_inventory['trays'][0]['trayId']
            self.current_configuration_cache['used_shelf_ids'] = [tray['trayId'] for tray in hardware_inventory['trays']]
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve controller shelf identifier. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
    return self.current_configuration_cache