from __future__ import (absolute_import, division, print_function)
import os
import subprocess
import sys
from ansible.module_utils.common.text.converters import to_bytes
import runpy
import sys
def respawn_module(interpreter_path):
    """
    Respawn the currently-running Ansible Python module under the specified Python interpreter.

    Ansible modules that require libraries that are typically available only under well-known interpreters
    (eg, ``yum``, ``apt``, ``dnf``) can use bespoke logic to determine the libraries they need are not
    available, then call `respawn_module` to re-execute the current module under a different interpreter
    and exit the current process when the new subprocess has completed. The respawned process inherits only
    stdout/stderr from the current process.

    Only a single respawn is allowed. ``respawn_module`` will fail on nested respawns. Modules are encouraged
    to call `has_respawned()` to defensively guide behavior before calling ``respawn_module``, and to ensure
    that the target interpreter exists, as ``respawn_module`` will not fail gracefully.

    :arg interpreter_path: path to a Python interpreter to respawn the current module
    """
    if has_respawned():
        raise Exception('module has already been respawned')
    payload = _create_payload()
    stdin_read, stdin_write = os.pipe()
    os.write(stdin_write, to_bytes(payload))
    os.close(stdin_write)
    rc = subprocess.call([interpreter_path, '--'], stdin=stdin_read)
    sys.exit(rc)