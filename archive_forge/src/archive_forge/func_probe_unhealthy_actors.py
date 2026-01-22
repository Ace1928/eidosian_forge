from collections import defaultdict
import copy
from dataclasses import dataclass
import logging
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError, RayError, RayTaskError
from ray.rllib.utils.typing import T
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def probe_unhealthy_actors(self, timeout_seconds: Optional[int]=None, mark_healthy: bool=False) -> List[int]:
    """Ping all unhealthy actors to try bringing them back.

        Args:
            timeout_seconds: Timeout to avoid pinging hanging workers indefinitely.
            mark_healthy: Whether to mark actors healthy if they respond to the ping.

        Returns:
            A list of actor ids that are restored.
        """
    unhealthy_actor_ids = [actor_id for actor_id in self.actor_ids() if not self.is_actor_healthy(actor_id)]
    if not unhealthy_actor_ids:
        return []
    remote_results = self.foreach_actor(func=lambda actor: actor.ping(), remote_actor_ids=unhealthy_actor_ids, healthy_only=False, timeout_seconds=timeout_seconds, mark_healthy=mark_healthy)
    return [result.actor_id for result in remote_results if result.ok]