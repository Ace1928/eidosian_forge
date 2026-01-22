from __future__ import (absolute_import, division, print_function)
import json  # noqa: F402
import os  # noqa: F402
import shlex  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import normalize_signal
from ansible_collections.containers.podman.plugins.module_utils.podman.common import ARGUMENTS_OPTS_DICT
def diffparam_ulimit(self):
    after = self.params['ulimit'] or []
    if 'createcommand' in self.info['config']:
        before = self._createcommand('--ulimit')
        before, after = (sorted(before), sorted(after))
        return self._diff_update_and_compare('ulimit', before, after)
    if after:
        ulimits = self.info['hostconfig']['ulimits']
        before = {u['name'].replace('rlimit_', ''): '%s:%s' % (u['soft'], u['hard']) for u in ulimits}
        after = {i.split('=')[0]: i.split('=')[1] for i in self.params['ulimit']}
        new_before = []
        new_after = []
        for u in list(after.keys()):
            if u in before and '-1' not in after[u]:
                new_before.append([u, before[u]])
                new_after.append([u, after[u]])
        return self._diff_update_and_compare('ulimit', new_before, new_after)
    return self._diff_update_and_compare('ulimit', '', '')