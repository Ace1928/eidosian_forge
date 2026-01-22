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
@mutex
def get_docker_preferred_network_name(args: EnvironmentConfig) -> t.Optional[str]:
    """
    Return the preferred network name for use with Docker. The selection logic is:
    - the network selected by the user with `--docker-network`
    - the network of the currently running docker container (if any)
    - the default docker network (returns None)
    """
    try:
        return get_docker_preferred_network_name.network
    except AttributeError:
        pass
    network = None
    if args.docker_network:
        network = args.docker_network
    else:
        current_container_id = get_docker_container_id()
        if current_container_id:
            container = docker_inspect(args, current_container_id, always=True)
            network = container.get_network_name()
    if network is None and require_docker().command == 'podman' and docker_network_inspect(args, 'podman', always=True):
        network = 'podman'
    get_docker_preferred_network_name.network = network
    return network