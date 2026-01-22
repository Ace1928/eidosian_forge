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
def get_image_info(self):
    """Inspect container image and gather info about it."""
    is_rootfs = self.module_params['rootfs']
    if is_rootfs:
        return {'Id': self.module_params['image']}
    rc, out, err = self.module.run_command([self.module_params['executable'], b'image', b'inspect', self.module_params['image']])
    return json.loads(out)[0] if rc == 0 else {}