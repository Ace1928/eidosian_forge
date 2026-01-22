import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def latency_graph(self):
    """Raise a NotImplementedError, as the client will not support LATENCY GRAPH.
        This funcion is best used within the redis-cli.

        For more information see https://redis.io/commands/latency-graph.
        """
    raise NotImplementedError('\n            LATENCY GRAPH is intentionally not implemented in the client.\n\n            For more information see https://redis.io/commands/latency-graph\n            ')