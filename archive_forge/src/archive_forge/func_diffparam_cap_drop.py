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
def diffparam_cap_drop(self):
    before = self.info['effectivecaps'] or []
    before = [i.lower() for i in before]
    after = before[:]
    if self.module_params['cap_drop'] is not None:
        for cap in self.module_params['cap_drop']:
            cap = cap.lower()
            cap = cap if cap.startswith('cap_') else 'cap_' + cap
            if cap in after:
                after.remove(cap)
    before, after = (sorted(list(set(before))), sorted(list(set(after))))
    return self._diff_update_and_compare('cap_drop', before, after)