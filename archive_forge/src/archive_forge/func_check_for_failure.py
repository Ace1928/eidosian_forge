import abc
import functools
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import StartTraceback, find_free_port
from ray.exceptions import RayActorError
from ray.types import ObjectRef
def check_for_failure(remote_values: List[ObjectRef]) -> Tuple[bool, Optional[Exception]]:
    """Check for actor failure when retrieving the remote values.

    Args:
        remote_values: List of object references from Ray actor methods.

    Returns:
        A tuple of (bool, Exception). The bool is
        True if evaluating all object references is successful, False otherwise.
    """
    unfinished = remote_values.copy()
    while len(unfinished) > 0:
        finished, unfinished = ray.wait(unfinished)
        for object_ref in finished:
            try:
                ray.get(object_ref)
            except RayActorError as exc:
                failed_actor_rank = remote_values.index(object_ref)
                logger.info(f'Worker {failed_actor_rank} has failed.')
                return (False, exc)
            except Exception as exc:
                raise StartTraceback from exc
    return (True, None)