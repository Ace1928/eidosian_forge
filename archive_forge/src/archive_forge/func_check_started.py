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
def check_started(self) -> Tuple[ReplicaStartupStatus, Optional[str]]:
    """Check if the replica has started. If so, transition to RUNNING.

        Should handle the case where the replica has already stopped.

        Returns:
            status: Most recent state of replica by
                querying actor obj ref
        """
    is_ready = self._actor.check_ready()
    self.update_actor_details(pid=self._actor.pid, node_id=self._actor.node_id, node_ip=self._actor.node_ip, actor_id=self._actor.actor_id, worker_id=self._actor.worker_id, log_file_path=self._actor.log_file_path)
    return is_ready