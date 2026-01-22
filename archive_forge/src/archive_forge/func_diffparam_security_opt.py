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
def diffparam_security_opt(self):
    unsorted_before = self.info['hostconfig']['securityopt']
    unsorted_after = self.params['security_opt']
    before = sorted((item for element in unsorted_before for item in element.split(',') if 'apparmor=container-default' not in item))
    after = sorted(list(set(unsorted_after)))
    return self._diff_update_and_compare('security_opt', before, after)