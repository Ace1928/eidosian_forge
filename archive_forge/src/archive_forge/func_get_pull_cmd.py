from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_cli import (
from ansible_collections.community.docker.plugins.module_utils.compose_v2 import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
def get_pull_cmd(self, dry_run, no_start=False):
    args = self.get_base_args() + ['pull']
    if self.policy != 'always':
        args.extend(['--policy', self.policy])
    if dry_run:
        args.append('--dry-run')
    args.append('--')
    return args