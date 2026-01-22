from __future__ import annotations
import collections.abc as c
import contextlib
import json
import random
import time
import uuid
import threading
import typing as t
from .util import (
from .util_common import (
from .config import (
from .docker_util import (
from .ansible_util import (
from .core_ci import (
from .target import (
from .ssh import (
from .host_configs import (
from .connections import (
from .thread import (
def run_container(args: EnvironmentConfig, image: str, name: str, options: t.Optional[list[str]], cmd: t.Optional[list[str]]=None, create_only: bool=False) -> str:
    """Run a container using the given docker image."""
    options = list(options or [])
    cmd = list(cmd or [])
    options.extend(['--name', name])
    network = get_docker_preferred_network_name(args)
    if is_docker_user_defined_network(network):
        options.extend(['--network', network])
    for _iteration in range(1, 3):
        try:
            if create_only:
                stdout = docker_create(args, image, options, cmd)[0]
            else:
                stdout = docker_run(args, image, options, cmd)[0]
        except SubprocessError as ex:
            display.error(ex.message)
            display.warning(f'Failed to run docker image "{image}". Waiting a few seconds before trying again.')
            docker_rm(args, name)
            time.sleep(3)
        else:
            if args.explain:
                stdout = ''.join((random.choice('0123456789abcdef') for _iteration in range(64)))
            return stdout.strip()
    raise ApplicationError(f'Failed to run docker image "{image}".')