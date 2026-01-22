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
def start_stop_delete(self):

    def complete_params(cmd):
        if self.params['attach'] and self.action == 'start':
            cmd.append('--attach')
        if self.params['detach'] is False and self.action == 'start' and ('--attach' not in cmd):
            cmd.append('--attach')
        if self.params['detach_keys'] and self.action == 'start':
            cmd += ['--detach-keys', self.params['detach_keys']]
        if self.params['sig_proxy'] and self.action == 'start':
            cmd.append('--sig-proxy')
        if self.params['stop_time'] and self.action == 'stop':
            cmd += ['--time', self.params['stop_time']]
        if self.params['restart_time'] and self.action == 'restart':
            cmd += ['--time', self.params['restart_time']]
        if self.params['delete_depend'] and self.action == 'delete':
            cmd.append('--depend')
        if self.params['delete_time'] and self.action == 'delete':
            cmd += ['--time', self.params['delete_time']]
        if self.params['delete_volumes'] and self.action == 'delete':
            cmd.append('--volumes')
        if self.params['force_delete'] and self.action == 'delete':
            cmd.append('--force')
        return cmd
    if self.action in ['stop', 'start', 'restart']:
        cmd = complete_params([self.action]) + [self.params['name']]
        return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]
    if self.action == 'delete':
        cmd = complete_params(['rm']) + [self.params['name']]
        return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]