import dataclasses
import inspect
import logging
from collections import defaultdict
from functools import wraps
from typing import List, Optional, Tuple
import aiohttp
import grpc
from grpc.aio._call import UnaryStreamCall
import ray
import ray.dashboard.modules.log.log_consts as log_consts
from ray._private import ray_constants
from ray._private.gcs_utils import GcsAioClient
from ray._private.utils import hex_to_binary
from ray._raylet import ActorID, JobID, TaskID
from ray.core.generated import gcs_service_pb2_grpc
from ray.core.generated.gcs_pb2 import ActorTableData
from ray.core.generated.gcs_service_pb2 import (
from ray.core.generated.node_manager_pb2 import (
from ray.core.generated.node_manager_pb2_grpc import NodeManagerServiceStub
from ray.core.generated.reporter_pb2 import (
from ray.core.generated.reporter_pb2_grpc import LogServiceStub
from ray.core.generated.runtime_env_agent_pb2 import (
from ray.dashboard.datacenter import DataSource
from ray.dashboard.modules.job.common import JobInfoStorageClient
from ray.dashboard.modules.job.pydantic_models import JobDetails, JobType
from ray.dashboard.modules.job.utils import get_driver_jobs
from ray.dashboard.utils import Dict as Dictionary
from ray.util.state.common import (
from ray.util.state.exception import DataSourceUnavailable
def register_agent_client(self, node_id, address: str, port: int):
    options = _STATE_MANAGER_GRPC_OPTIONS
    channel = ray._private.utils.init_grpc_channel(f'{address}:{port}', options=options, asynchronous=True)
    self._log_agent_stub[node_id] = LogServiceStub(channel)
    self._id_id_map.put(node_id, address)