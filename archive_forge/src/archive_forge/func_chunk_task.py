import math
import logging
import queue
import threading
import warnings
import grpc
from collections import OrderedDict
from typing import Any, Callable, Dict, TYPE_CHECKING, Optional, Union
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.client.common import (
from ray.util.debug import log_once
def chunk_task(req: ray_client_pb2.DataRequest):
    """
    Chunks a client task. Doing this lazily is important with large arguments,
    since taking slices of bytes objects does a copy. This means if we
    immediately materialized every chunk of a large argument and inserted them
    into the result_queue, we would effectively double the memory needed
    on the client to handle the task.
    """
    request_data = req.task.data
    total_size = len(request_data)
    assert total_size > 0, 'Cannot chunk object with missing data'
    total_chunks = math.ceil(total_size / OBJECT_TRANSFER_CHUNK_SIZE)
    for chunk_id in range(0, total_chunks):
        start = chunk_id * OBJECT_TRANSFER_CHUNK_SIZE
        end = min(total_size, (chunk_id + 1) * OBJECT_TRANSFER_CHUNK_SIZE)
        chunk = ray_client_pb2.ClientTask(type=req.task.type, name=req.task.name, payload_id=req.task.payload_id, client_id=req.task.client_id, options=req.task.options, baseline_options=req.task.baseline_options, namespace=req.task.namespace, data=request_data[start:end], chunk_id=chunk_id, total_chunks=total_chunks)
        yield ray_client_pb2.DataRequest(req_id=req.req_id, task=chunk)