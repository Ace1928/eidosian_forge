import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.util.annotations import PublicAPI
def set_default_actor_lifetime(self, default_actor_lifetime: str) -> None:
    """Set the default actor lifetime, which can be "detached" or "non_detached".

        See :ref:`actor lifetimes <actor-lifetimes>` for more details.

        Args:
            default_actor_lifetime: The default actor lifetime to set.
        """
    import ray.core.generated.common_pb2 as common_pb2
    if default_actor_lifetime == 'detached':
        self._default_actor_lifetime = common_pb2.JobConfig.ActorLifetime.DETACHED
    elif default_actor_lifetime == 'non_detached':
        self._default_actor_lifetime = common_pb2.JobConfig.ActorLifetime.NON_DETACHED
    else:
        raise ValueError('Default actor lifetime must be one of `detached`, `non_detached`')