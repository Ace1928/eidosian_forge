from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def make_update_body(self, target_iface):
    body = dict(iscsiInterface=target_iface['id'])
    update_required = False
    self._logger.info('Requested state=%s.', self.state)
    self._logger.info('config_method: current=%s, requested=%s', target_iface['ipv4Data']['ipv4AddressConfigMethod'], self.config_method)
    if self.state == 'enabled':
        settings = dict()
        if not target_iface['ipv4Enabled']:
            update_required = True
            settings['ipv4Enabled'] = [True]
        if self.mtu != target_iface['interfaceData']['ethernetData']['maximumFramePayloadSize']:
            update_required = True
            settings['maximumFramePayloadSize'] = [self.mtu]
        if self.config_method == 'static':
            ipv4Data = target_iface['ipv4Data']['ipv4AddressData']
            if ipv4Data['ipv4Address'] != self.address:
                update_required = True
                settings['ipv4Address'] = [self.address]
            if ipv4Data['ipv4SubnetMask'] != self.subnet_mask:
                update_required = True
                settings['ipv4SubnetMask'] = [self.subnet_mask]
            if self.gateway is not None and ipv4Data['ipv4GatewayAddress'] != self.gateway:
                update_required = True
                settings['ipv4GatewayAddress'] = [self.gateway]
            if target_iface['ipv4Data']['ipv4AddressConfigMethod'] != 'configStatic':
                update_required = True
                settings['ipv4AddressConfigMethod'] = ['configStatic']
        elif target_iface['ipv4Data']['ipv4AddressConfigMethod'] != 'configDhcp':
            update_required = True
            settings.update(dict(ipv4Enabled=[True], ipv4AddressConfigMethod=['configDhcp']))
        body['settings'] = settings
    elif target_iface['ipv4Enabled']:
        update_required = True
        body['settings'] = dict(ipv4Enabled=[False])
    self._logger.info('Update required ?=%s', update_required)
    self._logger.info('Update body: %s', pformat(body))
    return (update_required, body)