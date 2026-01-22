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
def create_support_container_context(args: EnvironmentConfig, ssh: t.Optional[SshConnectionDetail], containers: ContainerDatabase) -> SupportContainerContext:
    """Context manager that provides SSH port forwards. Returns updated container metadata."""
    host_type = HostType.control
    revised = ContainerDatabase(containers.data.copy())
    source = revised.data.pop(HostType.origin, None)
    container_map: dict[tuple[str, int], tuple[str, str, int]] = {}
    if host_type not in revised.data:
        if not source:
            raise Exception('Missing origin container details.')
        for context_name, context in source.items():
            for container_name, container in context.items():
                if '-controller-' in container_name:
                    continue
                for port, access_port in container.port_map():
                    container_map[container.host_ip, access_port] = (context_name, container_name, port)
    if not container_map:
        return SupportContainerContext(revised, None)
    if not ssh:
        raise Exception('The %s host was not pre-configured for container access and SSH forwarding is not available.' % host_type)
    forwards = list(container_map.keys())
    process = create_ssh_port_forwards(args, ssh, forwards)
    result = SupportContainerContext(revised, process)
    try:
        port_forwards = process.collect_port_forwards()
        contexts: dict[str, dict[str, ContainerAccess]] = {}
        for forward, forwarded_port in port_forwards.items():
            access_host, access_port = forward
            context_name, container_name, container_port = container_map[access_host, access_port]
            container = source[context_name][container_name]
            context = contexts.setdefault(context_name, {})
            forwarded_container = context.setdefault(container_name, ContainerAccess('127.0.0.1', container.names, None, {}))
            forwarded_container.forwards[container_port] = forwarded_port
            display.info('Container "%s" port %d available at %s:%d is forwarded over SSH as port %d.' % (container_name, container_port, access_host, access_port, forwarded_port), verbosity=1)
        revised.data[host_type] = contexts
        return result
    except Exception:
        result.close()
        raise