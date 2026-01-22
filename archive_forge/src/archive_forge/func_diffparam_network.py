from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def diffparam_network(self):
    if not self.infra_info:
        return self._diff_update_and_compare('network', [], [])
    net_mode_before = self.infra_info['hostconfig']['networkmode']
    net_mode_after = ''
    before = list(self.infra_info['networksettings'].get('networks', {}))
    if before == ['podman']:
        before = []
    after = self.params['network'] or []
    if net_mode_before == 'slirp4netns' and 'createcommand' in self.info:
        cr_com = self.info['createcommand']
        if '--network' in cr_com:
            cr_net = cr_com[cr_com.index('--network') + 1].lower()
            if 'slirp4netns:' in cr_net:
                before = [cr_net]
    if after in [['bridge'], ['host'], ['slirp4netns']]:
        net_mode_after = after[0]
    if net_mode_after and (not before):
        net_mode_after = net_mode_after.replace('bridge', 'default')
        net_mode_after = net_mode_after.replace('slirp4netns', 'default')
        net_mode_before = net_mode_before.replace('bridge', 'default')
        net_mode_before = net_mode_before.replace('slirp4netns', 'default')
        return self._diff_update_and_compare('network', net_mode_before, net_mode_after)
    if not net_mode_after and net_mode_before == 'slirp4netns' and (not after):
        net_mode_after = 'slirp4netns'
        if before == ['slirp4netns']:
            after = ['slirp4netns']
    if not net_mode_after and net_mode_before == 'bridge' and (not after):
        net_mode_after = 'bridge'
        if before == ['bridge']:
            after = ['bridge']
    before, after = (sorted(list(set(before))), sorted(list(set(after))))
    return self._diff_update_and_compare('network', before, after)