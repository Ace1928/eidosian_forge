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
def call_cli_json(self, *args, **kwargs):
    warn_on_stderr = kwargs.pop('warn_on_stderr', False)
    rc, stdout, stderr = self.call_cli(*args, **kwargs)
    if warn_on_stderr and stderr:
        self.warn(to_native(stderr))
    try:
        data = json.loads(stdout)
    except Exception as exc:
        self.fail('Error while parsing JSON output of {cmd}: {exc}\nJSON output: {stdout}'.format(cmd=self._compose_cmd_str(args), exc=to_native(exc), stdout=to_native(stdout)))
    return (rc, data, stderr)