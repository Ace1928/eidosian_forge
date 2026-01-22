from __future__ import (absolute_import, division, print_function)
from subprocess import Popen, PIPE
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.gnupg.cli import GPGError, GPGRunner

        Run ``[gpg] + command`` and return ``(rc, stdout, stderr)``.

        If ``data`` is not ``None``, it will be provided as stdin.
        The code assumes it is a bytes string.

        Returned stdout and stderr are native Python strings.
        Pass ``check_rc=False`` to allow return codes != 0.

        Raises a ``GPGError`` in case of errors.
        