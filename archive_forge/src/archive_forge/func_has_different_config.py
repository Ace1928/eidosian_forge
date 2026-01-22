from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
def has_different_config(self, net):
    """
        Evaluates an existing network and returns a tuple containing a boolean
        indicating if the configuration is different and a list of differences.

        :param net: the inspection output for an existing network
        :return: (bool, list)
        """
    differences = DifferenceTracker()
    if self.parameters.driver and self.parameters.driver != net['Driver']:
        differences.add('driver', parameter=self.parameters.driver, active=net['Driver'])
    if self.parameters.driver_options:
        if not net.get('Options'):
            differences.add('driver_options', parameter=self.parameters.driver_options, active=net.get('Options'))
        else:
            for key, value in self.parameters.driver_options.items():
                if not key in net['Options'] or value != net['Options'][key]:
                    differences.add('driver_options.%s' % key, parameter=value, active=net['Options'].get(key))
    if self.parameters.ipam_driver:
        if not net.get('IPAM') or net['IPAM']['Driver'] != self.parameters.ipam_driver:
            differences.add('ipam_driver', parameter=self.parameters.ipam_driver, active=net.get('IPAM'))
    if self.parameters.ipam_driver_options is not None:
        ipam_driver_options = net['IPAM'].get('Options') or {}
        if ipam_driver_options != self.parameters.ipam_driver_options:
            differences.add('ipam_driver_options', parameter=self.parameters.ipam_driver_options, active=ipam_driver_options)
    if self.parameters.ipam_config is not None and self.parameters.ipam_config:
        if not net.get('IPAM') or not net['IPAM']['Config']:
            differences.add('ipam_config', parameter=self.parameters.ipam_config, active=net.get('IPAM', {}).get('Config'))
        else:
            net_ipam_configs = []
            for net_ipam_config in net['IPAM']['Config']:
                config = dict()
                for k, v in net_ipam_config.items():
                    config[normalize_ipam_config_key(k)] = v
                net_ipam_configs.append(config)
            for idx, ipam_config in enumerate(self.parameters.ipam_config):
                net_config = dict()
                for net_ipam_config in net_ipam_configs:
                    if dicts_are_essentially_equal(ipam_config, net_ipam_config):
                        net_config = net_ipam_config
                        break
                for key, value in ipam_config.items():
                    if value is None:
                        continue
                    if value != net_config.get(key):
                        differences.add('ipam_config[%s].%s' % (idx, key), parameter=value, active=net_config.get(key))
    if self.parameters.enable_ipv6 is not None and self.parameters.enable_ipv6 != net.get('EnableIPv6', False):
        differences.add('enable_ipv6', parameter=self.parameters.enable_ipv6, active=net.get('EnableIPv6', False))
    if self.parameters.internal is not None and self.parameters.internal != net.get('Internal', False):
        differences.add('internal', parameter=self.parameters.internal, active=net.get('Internal'))
    if self.parameters.scope is not None and self.parameters.scope != net.get('Scope'):
        differences.add('scope', parameter=self.parameters.scope, active=net.get('Scope'))
    if self.parameters.attachable is not None and self.parameters.attachable != net.get('Attachable', False):
        differences.add('attachable', parameter=self.parameters.attachable, active=net.get('Attachable'))
    if self.parameters.labels:
        if not net.get('Labels'):
            differences.add('labels', parameter=self.parameters.labels, active=net.get('Labels'))
        else:
            for key, value in self.parameters.labels.items():
                if not key in net['Labels'] or value != net['Labels'][key]:
                    differences.add('labels.%s' % key, parameter=value, active=net['Labels'].get(key))
    return (not differences.empty, differences)