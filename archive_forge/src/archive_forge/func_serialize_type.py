import base64
from collections import OrderedDict
import importlib
import io
import zlib
from typing import Any, Dict, Optional, Sequence, Type, Union
import numpy as np
import ray
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.gym import try_import_gymnasium_and_gym
from ray.rllib.utils.error import NotSerializable
from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.spaces.simplex import Simplex
@DeveloperAPI
def serialize_type(type_: Union[Type, str]) -> str:
    """Converts a type into its full classpath ([module file] + "." + [class name]).

    Args:
        type_: The type to convert.

    Returns:
        The full classpath of the given type, e.g. "ray.rllib.algorithms.ppo.PPOConfig".
    """
    if isinstance(type_, str):
        return type_
    return type_.__module__ + '.' + type_.__qualname__