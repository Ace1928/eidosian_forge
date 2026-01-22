import base64
import functools
import gc
import inspect
import json
import logging
import math
import pickle
import queue
import threading
import time
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Dict, List, Optional, Set, Union
import grpc
import ray
import ray._private.state
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray import cloudpickle
from ray._private import ray_constants
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.ray_constants import env_integer
from ray._private.ray_logging import setup_logger
from ray._private.services import canonicalize_bootstrap_address_or_die
from ray._private.tls_utils import add_port_to_grpc_server
from ray.job_config import JobConfig
from ray.util.client.common import (
from ray.util.client.server.dataservicer import DataServicer
from ray.util.client.server.logservicer import LogstreamServicer
from ray.util.client.server.proxier import serve_proxier
from ray.util.client.server.server_pickler import dumps_from_server, loads_from_client
from ray.util.client.server.server_stubs import current_server
def send_get_response(result: Any) -> None:
    """Pushes GetResponses to the main DataPath loop to send
                    to the client. This is called when the object is ready
                    on the server side."""
    try:
        serialized = dumps_from_server(result, client_id, self)
        total_size = len(serialized)
        assert total_size > 0, 'Serialized object cannot be zero bytes'
        total_chunks = math.ceil(total_size / OBJECT_TRANSFER_CHUNK_SIZE)
        for chunk_id in range(request.start_chunk_id, total_chunks):
            start = chunk_id * OBJECT_TRANSFER_CHUNK_SIZE
            end = min(total_size, (chunk_id + 1) * OBJECT_TRANSFER_CHUNK_SIZE)
            get_resp = ray_client_pb2.GetResponse(valid=True, data=serialized[start:end], chunk_id=chunk_id, total_chunks=total_chunks, total_size=total_size)
            chunk_resp = ray_client_pb2.DataResponse(get=get_resp, req_id=req_id)
            result_queue.put(chunk_resp)
    except Exception as exc:
        get_resp = ray_client_pb2.GetResponse(valid=False, error=cloudpickle.dumps(exc))
        resp = ray_client_pb2.DataResponse(get=get_resp, req_id=req_id)
        result_queue.put(resp)