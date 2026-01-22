from ansible.errors import AnsibleError
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.urls import open_url
import contextlib
import os
import subprocess
import sys
import typing as t
from dataclasses import dataclass, fields as dc_fields
from functools import partial
from urllib.error import HTTPError, URLError
def run_gpg_verify(manifest_file, signature, keyring, display):
    status_fd_read, status_fd_write = os.pipe()
    remove_keybox = not os.path.exists(keyring)
    cmd = ['gpg', f'--status-fd={status_fd_write}', '--verify', '--batch', '--no-tty', '--no-default-keyring', f'--keyring={keyring}', '-', manifest_file]
    cmd_str = ' '.join(cmd)
    display.vvvv(f"Running command '{cmd}'")
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(status_fd_write,), encoding='utf8')
    except (FileNotFoundError, subprocess.SubprocessError) as err:
        raise AnsibleError(f"Failed during GnuPG verification with command '{cmd_str}': {err}") from err
    else:
        stdout, stderr = p.communicate(input=signature)
    finally:
        os.close(status_fd_write)
    if remove_keybox:
        with contextlib.suppress(OSError):
            os.remove(keyring)
    with os.fdopen(status_fd_read) as f:
        stdout = f.read()
        display.vvvv(f'stdout: \n{stdout}\nstderr: \n{stderr}\n(exit code {p.returncode})')
        return (stdout, p.returncode)