from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
def is_different(self):
    diff_func_list = [func for func in dir(self) if callable(getattr(self, func)) and func.startswith('diffparam')]
    fail_fast = not bool(self.module._diff)
    different = False
    for func_name in diff_func_list:
        dff_func = getattr(self, func_name)
        if dff_func():
            if fail_fast:
                return True
            different = True
    for p in self.non_idempotent:
        if self.module_params[p] is not None and self.module_params[p] not in [{}, [], '']:
            different = True
    return different