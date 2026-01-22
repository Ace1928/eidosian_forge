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
def update_replicas(self, replicas: List[ReplicaWrapper]):
    """Update the set of available replicas to be considered for scheduling.

        When the set of replicas changes, we may spawn additional scheduling tasks
        if there are pending requests.
        """
    new_replicas = {}
    new_replica_id_set = set()
    new_colocated_replica_ids = defaultdict(set)
    new_multiplexed_model_id_to_replica_ids = defaultdict(set)
    for r in replicas:
        new_replicas[r.replica_id] = r
        new_replica_id_set.add(r.replica_id)
        if self._self_node_id is not None and r.node_id == self._self_node_id:
            new_colocated_replica_ids[LocalityScope.NODE].add(r.replica_id)
        if self._self_availability_zone is not None and r.availability_zone == self._self_availability_zone:
            new_colocated_replica_ids[LocalityScope.AVAILABILITY_ZONE].add(r.replica_id)
        for model_id in r.multiplexed_model_ids:
            new_multiplexed_model_id_to_replica_ids[model_id].add(r.replica_id)
    if self._replica_id_set != new_replica_id_set:
        app_msg = f" in application '{self.app_name}'" if self.app_name else ''
        logger.info(f"Got updated replicas for deployment '{self._deployment_id.name}'{app_msg}: {new_replica_id_set}.", extra={'log_to_stderr': False})
    self._replicas = new_replicas
    self._replica_id_set = new_replica_id_set
    self._colocated_replica_ids = new_colocated_replica_ids
    self._multiplexed_model_id_to_replica_ids = new_multiplexed_model_id_to_replica_ids
    self._replicas_updated_event.set()
    self.maybe_start_scheduling_tasks()