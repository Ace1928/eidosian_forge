from collections import defaultdict, deque
import logging
import platform
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
import ray
from ray.actor import ActorClass, ActorHandle
def try_create_colocated(cls: Type[ActorClass], args: List[Any], count: int, kwargs: Optional[List[Any]]=None, node: Optional[str]='localhost') -> List[ActorHandle]:
    """Tries to co-locate (same node) a set of Actors of the same type.

    Returns a list of successfully co-located actors. All actors that could
    not be co-located (with the others on the given node) will not be in this
    list.

    Creates each actor via it's remote() constructor and then checks, whether
    it has been co-located (on the same node) with the other (already created)
    ones. If not, terminates the just created actor.

    Args:
        cls: The Actor class to use (already @ray.remote "converted").
        args: List of args to pass to the Actor's constructor. One item
            per to-be-created actor (`count`).
        count: Number of actors of the given `cls` to construct.
        kwargs: Optional list of kwargs to pass to the Actor's constructor.
            One item per to-be-created actor (`count`).
        node: The node to co-locate the actors on. By default ("localhost"),
            place the actors on the node the caller of this function is
            located on. If None, will try to co-locate all actors on
            any available node.

    Returns:
        List containing all successfully co-located actor handles.
    """
    if node == 'localhost':
        node = platform.node()
    kwargs = kwargs or {}
    actors = [cls.remote(*args, **kwargs) for _ in range(count)]
    co_located, non_co_located = split_colocated(actors, node=node)
    logger.info('Got {} colocated actors of {}'.format(len(co_located), count))
    for a in non_co_located:
        a.__ray_terminate__.remote()
    return co_located