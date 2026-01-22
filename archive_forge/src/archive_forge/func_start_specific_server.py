import atexit
import json
import logging
import socket
import sys
import time
import traceback
from concurrent import futures
from dataclasses import dataclass
from itertools import chain
import urllib
from threading import Event, Lock, RLock, Thread
from typing import Callable, Dict, List, Optional, Tuple
import grpc
import psutil
import ray
import ray.core.generated.agent_manager_pb2 as agent_manager_pb2
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
import ray.core.generated.runtime_env_agent_pb2 as runtime_env_agent_pb2
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.parameter import RayParams
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.services import ProcessInfo, start_ray_client_server
from ray._private.tls_utils import add_port_to_grpc_server
from ray._private.utils import detect_fate_sharing_support
from ray.cloudpickle.compat import pickle
from ray.job_config import JobConfig
from ray.util.client.common import (
from ray.util.client.server.dataservicer import _get_reconnecting_from_context
def start_specific_server(self, client_id: str, job_config: JobConfig) -> bool:
    """
        Start up a RayClient Server for an incoming client to
        communicate with. Returns whether creation was successful.
        """
    specific_server = self._get_server_for_client(client_id)
    assert specific_server, f'Server has not been created for: {client_id}'
    output, error = self.node.get_log_file_handles(f'ray_client_server_{specific_server.port}', unique=True)
    serialized_runtime_env = job_config._get_serialized_runtime_env()
    runtime_env_config = job_config._get_proto_runtime_env_config()
    if not serialized_runtime_env or serialized_runtime_env == '{}':
        serialized_runtime_env_context = RuntimeEnvContext().serialize()
    else:
        serialized_runtime_env_context = self._create_runtime_env(serialized_runtime_env=serialized_runtime_env, runtime_env_config=runtime_env_config, specific_server=specific_server)
    proc = start_ray_client_server(self.address, self.node.node_ip_address, specific_server.port, stdout_file=output, stderr_file=error, fate_share=self.fate_share, server_type='specific-server', serialized_runtime_env_context=serialized_runtime_env_context, redis_password=self._redis_password)
    pid = proc.process.pid
    if sys.platform != 'win32':
        psutil_proc = psutil.Process(pid)
    else:
        psutil_proc = None
    while psutil_proc is not None:
        if proc.process.poll() is not None:
            logger.error(f'SpecificServer startup failed for client: {client_id}')
            break
        cmd = psutil_proc.cmdline()
        if _match_running_client_server(cmd):
            break
        logger.debug('Waiting for Process to reach the actual client server.')
        time.sleep(0.5)
    specific_server.set_result(proc)
    logger.info(f'SpecificServer started on port: {specific_server.port} with PID: {pid} for client: {client_id}')
    return proc.process.poll() is None