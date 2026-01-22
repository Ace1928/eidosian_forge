from __future__ import (absolute_import, division, print_function)
import base64
import datetime
import io
import json
import os
import os.path
import shutil
import stat
import tarfile
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, NotFound
def determine_user_group(client, container, log=None):
    dummy, stdout, stderr = _execute_command(client, container, ['/bin/sh', '-c', 'id -u && id -g'], check_rc=True, log=log)
    stdout_lines = stdout.splitlines()
    if len(stdout_lines) != 2:
        raise DockerUnexpectedError('Expected two-line output to obtain user and group ID for container {container}, but got {lc} lines:\n{stdout}'.format(container=container, lc=len(stdout_lines), stdout=stdout))
    user_id, group_id = stdout_lines
    try:
        return (int(user_id), int(group_id))
    except ValueError:
        raise DockerUnexpectedError('Expected two-line output with numeric IDs to obtain user and group ID for container {container}, but got "{l1}" and "{l2}" instead'.format(container=container, l1=user_id, l2=group_id))