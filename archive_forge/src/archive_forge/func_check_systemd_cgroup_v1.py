from __future__ import annotations
import abc
import dataclasses
import os
import shlex
import tempfile
import time
import typing as t
from .io import (
from .config import (
from .host_configs import (
from .core_ci import (
from .util import (
from .util_common import (
from .docker_util import (
from .bootstrap import (
from .venv import (
from .ssh import (
from .ansible_util import (
from .containers import (
from .connections import (
from .become import (
from .completion import (
from .dev.container_probe import (
def check_systemd_cgroup_v1(self, options: list[str]) -> None:
    """Check the cgroup v1 systemd hierarchy to verify it is writeable for our container."""
    probe_script = read_text_file(os.path.join(ANSIBLE_TEST_TARGET_ROOT, 'setup', 'check_systemd_cgroup_v1.sh')).replace('@MARKER@', self.MARKER).replace('@LABEL@', f'{self.label}-{self.args.session_name}')
    cmd = ['sh']
    try:
        run_utility_container(self.args, f'ansible-test-cgroup-check-{self.label}', cmd, options, data=probe_script)
    except SubprocessError as ex:
        if (error := self.extract_error(ex.stderr)):
            raise ControlGroupError(self.args, f'Unable to create a v1 cgroup within the systemd hierarchy.\nReason: {error}') from ex
        raise