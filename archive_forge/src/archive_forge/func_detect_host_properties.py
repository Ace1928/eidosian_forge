from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import re
import socket
import time
import urllib.parse
import typing as t
from .util import (
from .util_common import (
from .config import (
from .thread import (
from .cgroup import (
@mutex
def detect_host_properties(args: CommonConfig) -> ContainerHostProperties:
    """
    Detect and return properties of the container host.

    The information collected is:

      - The errno result from attempting to query the container host's audit status.
      - The max number of open files supported by the container host to run containers.
        This value may be capped to the maximum value used by ansible-test.
        If the value is below the desired limit, a warning is displayed.
      - The loginuid used by the container host to run containers, or None if the audit subsystem is unavailable.
      - The cgroup subsystems registered with the Linux kernel.
      - The mounts visible within a container.
      - The status of the systemd cgroup v1 hierarchy.

    This information is collected together to reduce the number of container runs to probe the container host.
    """
    try:
        return detect_host_properties.properties
    except AttributeError:
        pass
    single_line_commands = ('audit-status', 'cat /proc/sys/fs/nr_open', 'ulimit -Hn', '(cat /proc/1/loginuid; echo)')
    multi_line_commands = (' && '.join(single_line_commands), 'cat /proc/1/cgroup', 'cat /proc/1/mountinfo')
    options = ['--volume', '/sys/fs/cgroup:/probe:ro']
    cmd = ['sh', '-c', ' && echo "-" && '.join(multi_line_commands)]
    stdout = run_utility_container(args, 'ansible-test-probe', cmd, options)[0]
    if args.explain:
        return ContainerHostProperties(audit_code='???', max_open_files=MAX_NUM_OPEN_FILES, loginuid=LOGINUID_NOT_SET, cgroup_v1=SystemdControlGroupV1Status.VALID, cgroup_v2=False)
    blocks = stdout.split('\n-\n')
    values = blocks[0].split('\n')
    audit_parts = values[0].split(' ', 1)
    audit_status = int(audit_parts[0])
    audit_code = audit_parts[1]
    system_limit = int(values[1])
    hard_limit = int(values[2])
    loginuid = int(values[3]) if values[3] else None
    cgroups = CGroupEntry.loads(blocks[1])
    mounts = MountEntry.loads(blocks[2])
    if hard_limit < MAX_NUM_OPEN_FILES and hard_limit < system_limit and (require_docker().command == 'docker'):
        options = ['--ulimit', f'nofile={min(system_limit, MAX_NUM_OPEN_FILES)}']
        cmd = ['sh', '-c', 'ulimit -Hn']
        try:
            stdout = run_utility_container(args, 'ansible-test-ulimit', cmd, options)[0]
        except SubprocessError as ex:
            display.warning(str(ex))
        else:
            hard_limit = int(stdout)
    subsystems = set((cgroup.subsystem for cgroup in cgroups))
    mount_types = {mount.path: mount.type for mount in mounts}
    if 'systemd' not in subsystems:
        cgroup_v1 = SystemdControlGroupV1Status.SUBSYSTEM_MISSING
    elif not (mount_type := mount_types.get(pathlib.PurePosixPath('/probe/systemd'))):
        cgroup_v1 = SystemdControlGroupV1Status.FILESYSTEM_NOT_MOUNTED
    elif mount_type != MountType.CGROUP_V1:
        cgroup_v1 = SystemdControlGroupV1Status.MOUNT_TYPE_NOT_CORRECT
    else:
        cgroup_v1 = SystemdControlGroupV1Status.VALID
    cgroup_v2 = mount_types.get(pathlib.PurePosixPath('/probe')) == MountType.CGROUP_V2
    display.info(f'Container host audit status: {audit_code} ({audit_status})', verbosity=1)
    display.info(f'Container host max open files: {hard_limit}', verbosity=1)
    display.info(f'Container loginuid: {(loginuid if loginuid is not None else 'unavailable')}{(' (not set)' if loginuid == LOGINUID_NOT_SET else '')}', verbosity=1)
    if hard_limit < MAX_NUM_OPEN_FILES:
        display.warning(f'Unable to set container max open files to {MAX_NUM_OPEN_FILES}. Using container host limit of {hard_limit} instead.')
    else:
        hard_limit = MAX_NUM_OPEN_FILES
    properties = ContainerHostProperties(audit_code=audit_code, max_open_files=hard_limit, loginuid=loginuid, cgroup_v1=cgroup_v1, cgroup_v2=cgroup_v2)
    detect_host_properties.properties = properties
    return properties