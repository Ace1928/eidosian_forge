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
def diffparam_log_opt(self):
    before, after = ({}, {})
    path_before = None
    if 'logpath' in self.info:
        path_before = self.info['logpath']
    if 'logconfig' in self.info['hostconfig'] and 'path' in self.info['hostconfig']['logconfig']:
        path_before = self.info['hostconfig']['logconfig']['path']
    if path_before is not None:
        if self.module_params['log_opt'] and 'path' in self.module_params['log_opt'] and (self.module_params['log_opt']['path'] is not None):
            path_after = self.params['log_opt']['path']
        else:
            path_after = path_before
        if path_before != path_after:
            before.update({'log-path': path_before})
            after.update({'log-path': path_after})
    tag_before = None
    if 'logtag' in self.info:
        tag_before = self.info['logtag']
    if 'logconfig' in self.info['hostconfig'] and 'tag' in self.info['hostconfig']['logconfig']:
        tag_before = self.info['hostconfig']['logconfig']['tag']
    if tag_before is not None:
        if self.module_params['log_opt'] and 'tag' in self.module_params['log_opt'] and (self.module_params['log_opt']['tag'] is not None):
            tag_after = self.params['log_opt']['tag']
        else:
            tag_after = ''
        if tag_before != tag_after:
            before.update({'log-tag': tag_before})
            after.update({'log-tag': tag_after})
    return self._diff_update_and_compare('log_opt', before, after)