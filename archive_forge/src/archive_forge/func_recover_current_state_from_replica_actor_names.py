import json
import logging
import math
import os
import random
import time
import traceback
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import ObjectRef, cloudpickle
from ray.actor import ActorHandle
from ray.exceptions import RayActorError, RayError, RayTaskError, RuntimeEnvSetupError
from ray.serve import metrics
from ray.serve._private import default_impl
from ray.serve._private.autoscaling_metrics import InMemoryMetricsStore
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_scheduler import (
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve._private.version import DeploymentVersion, VersionedReplica
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.schema import (
from ray.util.placement_group import PlacementGroup
def recover_current_state_from_replica_actor_names(self, replica_actor_names: List[str]):
    """Recover deployment state from live replica actors found in the cluster."""
    assert self._target_state is not None, 'Target state should be recovered successfully first before recovering current state from replica actor names.'
    logger.info(f"Recovering current state for deployment '{self.deployment_name}' in application '{self.app_name}' from {len(replica_actor_names)} total actors.")
    for replica_actor_name in replica_actor_names:
        replica_name: ReplicaName = ReplicaName.from_str(replica_actor_name)
        new_deployment_replica = DeploymentReplica(self._controller_name, replica_name.replica_tag, replica_name.deployment_id, self._target_state.version)
        new_deployment_replica.recover()
        self._replicas.add(ReplicaState.RECOVERING, new_deployment_replica)
        self._deployment_scheduler.on_replica_recovering(replica_name.deployment_id, replica_name.replica_tag)
        logger.debug(f'RECOVERING replica: {new_deployment_replica.replica_tag}, deployment: {self.deployment_name}, application: {self.app_name}.')