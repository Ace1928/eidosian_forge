import asyncio
import enum
import logging
import math
import pickle
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
import ray
from ray._private.utils import load_class
from ray.actor import ActorHandle
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.exceptions import RayActorError
from ray.serve._private.common import DeploymentID, RequestProtocol, RunningReplicaInfo
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.long_poll import LongPollClient, LongPollNamespace
from ray.serve._private.utils import JavaActorHandleProxy, MetricsPusher
from ray.serve.generated.serve_pb2 import DeploymentRoute
from ray.serve.generated.serve_pb2 import RequestMetadata as RequestMetadataProto
from ray.serve.grpc_util import RayServegRPCContext
from ray.util import metrics
def fulfill_next_pending_request(self, replica: ReplicaWrapper, request_metadata: Optional[RequestMetadata]=None):
    """Assign the replica to the next pending request in FIFO order.

        If a pending request has been cancelled, it will be popped from the queue
        and not assigned.
        """
    matched_pending_request = self._get_pending_request_matching_metadata(replica, request_metadata)
    if matched_pending_request is not None:
        matched_pending_request.future.set_result(replica)
        self._pending_requests_to_fulfill.remove(matched_pending_request)
        return
    while len(self._pending_requests_to_fulfill) > 0:
        pr = self._pending_requests_to_fulfill.popleft()
        if not pr.future.done():
            pr.future.set_result(replica)
            break