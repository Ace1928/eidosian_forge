from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def update_networks(self, container, container_created):
    updated_container = container
    if self.all_options['networks'].comparison != 'ignore' or container_created:
        has_network_differences, network_differences = self.has_network_differences(container)
        if has_network_differences:
            if self.diff.get('differences'):
                self.diff['differences'].append(dict(network_differences=network_differences))
            else:
                self.diff['differences'] = [dict(network_differences=network_differences)]
            for netdiff in network_differences:
                self.diff_tracker.add('network.{0}'.format(netdiff['parameter']['name']), parameter=netdiff['parameter'], active=netdiff['container'])
            self.results['changed'] = True
            updated_container = self._add_networks(container, network_differences)
    purge_networks = self.all_options['networks'].comparison == 'strict' and self.module.params['networks'] is not None
    if not purge_networks and self.module.params['purge_networks']:
        purge_networks = True
        self.module.deprecate('The purge_networks option is used while networks is not specified. In this case purge_networks=true cannot be replaced by `networks: strict` in comparisons, which is necessary once purge_networks is removed. Please modify the docker_container invocation by adding `networks: []`', version='4.0.0', collection_name='community.docker')
    if purge_networks:
        has_extra_networks, extra_networks = self.has_extra_networks(container)
        if has_extra_networks:
            if self.diff.get('differences'):
                self.diff['differences'].append(dict(purge_networks=extra_networks))
            else:
                self.diff['differences'] = [dict(purge_networks=extra_networks)]
            for extra_network in extra_networks:
                self.diff_tracker.add('network.{0}'.format(extra_network['name']), active=extra_network)
            self.results['changed'] = True
            updated_container = self._purge_networks(container, extra_networks)
    return updated_container