from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def diffparam_publish(self):

    def compose(p, h):
        s = ':'.join([str(h['hostport']), p.replace('/tcp', '')]).strip(':')
        if h['hostip']:
            return ':'.join([h['hostip'], s])
        return s
    if not self.infra_info:
        return self._diff_update_and_compare('publish', '', '')
    ports = self.infra_info['hostconfig']['portbindings']
    before = []
    for port, hosts in ports.items():
        if hosts:
            for h in hosts:
                before.append(compose(port, h))
    after = self.params['publish'] or []
    after = [i.replace('/tcp', '').replace('[', '').replace(']', '') for i in after]
    for ports in after:
        if '-' in ports:
            return self._diff_update_and_compare('publish', '', '')
    before, after = (sorted(list(set(before))), sorted(list(set(after))))
    return self._diff_update_and_compare('publish', before, after)