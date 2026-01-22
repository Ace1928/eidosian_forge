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
def get_deployment_details(self, id: DeploymentID) -> Optional[DeploymentDetails]:
    """Gets detailed info on a deployment.

        Returns:
            DeploymentDetails: if the deployment is live.
            None: if the deployment is deleted.
        """
    statuses = self.get_deployment_statuses([id])
    if len(statuses) == 0:
        return None
    else:
        status_info = statuses[0]
        return DeploymentDetails(name=id.name, status=status_info.status, status_trigger=status_info.status_trigger, message=status_info.message, deployment_config=_deployment_info_to_schema(id.name, self.get_deployment(id)), replicas=self._deployment_states[id].list_replica_details())