from typing import Any, Callable, Type
import numpy as np
import tree  # dm_tree
from ray.rllib.connectors.connector import (
from ray.rllib.connectors.registry import register_connector
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
A util to register any simple transforming function as an AgentConnector

    The only requirement is that fn should take a single data object and return
    a single data object.

    Args:
        name: Name of the resulting actor connector.
        fn: The function that transforms env / agent data.

    Returns:
        A new AgentConnector class that transforms data using fn.
    