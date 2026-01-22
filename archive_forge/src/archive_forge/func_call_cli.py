from __future__ import (absolute_import, division, print_function)
import abc
import json
import shlex
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.auth import resolve_repository_name
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
def call_cli(self, *args, **kwargs):
    check_rc = kwargs.pop('check_rc', False)
    data = kwargs.pop('data', None)
    cwd = kwargs.pop('cwd', None)
    environ_update = kwargs.pop('environ_update', None)
    if kwargs:
        raise TypeError("call_cli() got an unexpected keyword argument '%s'" % list(kwargs)[0])
    environment = self._environment.copy()
    if environ_update:
        environment.update(environ_update)
    rc, stdout, stderr = self.module.run_command(self._compose_cmd(args), binary_data=True, check_rc=check_rc, cwd=cwd, data=data, encoding=None, environ_update=environment, expand_user_and_vars=False, ignore_invalid_cwd=False)
    return (rc, stdout, stderr)