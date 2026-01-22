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
def is_actor_healthy(self, actor_id: int) -> bool:
    """Whether a remote actor is in healthy state.

        Args:
            actor_id: ID of the remote actor.

        Returns:
            True if the actor is healthy, False otherwise.
        """
    if actor_id not in self.__remote_actor_states:
        raise ValueError(f'Unknown actor id: {actor_id}')
    return self.__remote_actor_states[actor_id].is_healthy