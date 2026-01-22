import asyncio
from datetime import datetime
import inspect
import fnmatch
import functools
import io
import json
import logging
import math
import os
import pathlib
import random
import socket
import subprocess
import sys
import tempfile
import time
import timeit
import traceback
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional
import uuid
from dataclasses import dataclass
import requests
from ray._raylet import Config
import psutil  # We must import psutil after ray because we bundle it with ray.
from ray._private import (
from ray._private.worker import RayContext
import yaml
import ray
import ray._private.gcs_utils as gcs_utils
import ray._private.memory_monitor as memory_monitor
import ray._private.services
import ray._private.utils
from ray._private.internal_api import memory_summary
from ray._private.tls_utils import generate_self_signed_tls_certs
from ray._raylet import GcsClientOptions, GlobalStateAccessor
from ray.core.generated import (
from ray.util.queue import Empty, Queue, _QueueActor
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def start_redis_instance(session_dir_path: str, port: int, redis_max_clients: Optional[int]=None, num_retries: int=20, stdout_file: Optional[str]=None, stderr_file: Optional[str]=None, password: Optional[str]=None, redis_max_memory: Optional[int]=None, fate_share: Optional[bool]=None, port_denylist: Optional[List[int]]=None, listen_to_localhost_only: bool=False, enable_tls: bool=False, replica_of=None, leader_id=None, db_dir=None, free_port=0):
    """Start a single Redis server.

    Notes:
        We will initially try to start the Redis instance at the given port,
        and then try at most `num_retries - 1` times to start the Redis
        instance at successive random ports.

    Args:
        session_dir_path: Path to the session directory of
            this Ray cluster.
        port: Try to start a Redis server at this port.
        redis_max_clients: If this is provided, Ray will attempt to configure
            Redis with this maxclients number.
        num_retries: The number of times to attempt to start Redis at
            successive ports.
        stdout_file: A file handle opened for writing to redirect stdout to. If
            no redirection should happen, then this should be None.
        stderr_file: A file handle opened for writing to redirect stderr to. If
            no redirection should happen, then this should be None.
        password: Prevents external clients without the password
            from connecting to Redis if provided.
        redis_max_memory: The max amount of memory (in bytes) to allow redis
            to use, or None for no limit. Once the limit is exceeded, redis
            will start LRU eviction of entries.
        port_denylist: A set of denylist ports that shouldn't
            be used when allocating a new port.
        listen_to_localhost_only: Redis server only listens to
            localhost (127.0.0.1) if it's true,
            otherwise it listens to all network interfaces.
        enable_tls: Enable the TLS/SSL in Redis or not

    Returns:
        A tuple of the port used by Redis and ProcessInfo for the process that
            was started. If a port is passed in, then the returned port value
            is the same.

    Raises:
        Exception: An exception is raised if Redis could not be started.
    """
    assert os.path.isfile(REDIS_EXECUTABLE)
    command = [REDIS_EXECUTABLE]
    if password:
        if ' ' in password:
            raise ValueError('Spaces not permitted in redis password.')
        command += ['--requirepass', password]
    if redis_replicas() > 1:
        command += ['--cluster-enabled', 'yes', '--cluster-config-file', f'node-{port}']
    if enable_tls:
        command += ['--tls-port', str(port), '--loglevel', 'warning', '--port', str(free_port)]
    else:
        command += ['--port', str(port), '--loglevel', 'warning']
    if listen_to_localhost_only:
        command += ['--bind', '127.0.0.1']
    pidfile = os.path.join(session_dir_path, 'redis-' + uuid.uuid4().hex + '.pid')
    command += ['--pidfile', pidfile]
    if enable_tls:
        if Config.REDIS_CA_CERT():
            command += ['--tls-ca-cert-file', Config.REDIS_CA_CERT()]
        if Config.REDIS_CLIENT_CERT():
            command += ['--tls-cert-file', Config.REDIS_CLIENT_CERT()]
        if Config.REDIS_CLIENT_KEY():
            command += ['--tls-key-file', Config.REDIS_CLIENT_KEY()]
        if replica_of is not None:
            command += ['--tls-replication', 'yes']
        command += ['--tls-auth-clients', 'no', '--tls-cluster', 'yes']
    if sys.platform != 'win32':
        command += ['--save', '', '--appendonly', 'no']
    if db_dir is not None:
        command += ['--dir', str(db_dir)]
    process_info = ray._private.services.start_ray_process(command, ray_constants.PROCESS_TYPE_REDIS_SERVER, stdout_file=stdout_file, stderr_file=stderr_file, fate_share=fate_share)
    node_id = None
    if redis_replicas() > 1:
        import redis
        while True:
            try:
                redis_cli = get_redis_cli(port, enable_tls)
                if replica_of is None:
                    slots = [str(i) for i in range(16384)]
                    redis_cli.cluster('addslots', *slots)
                else:
                    print(redis_cli.cluster('meet', '127.0.0.1', str(replica_of)))
                    print(redis_cli.cluster('replicate', leader_id))
                node_id = redis_cli.cluster('myid')
                break
            except (redis.exceptions.ConnectionError, redis.exceptions.ResponseError) as e:
                from time import sleep
                print(f'Waiting for redis to be up {e} ')
                sleep(0.1)
    return (node_id, process_info)