import dis
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from typing import Optional, Callable
import ray
import ray._private.profiling as profiling
from ray import cloudpickle as pickle
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.utils import (
from ray._private.serialization import pickle_dumps
from ray._raylet import (
def compute_collision_identifier(self, function_or_class):
    """The identifier is used to detect excessive duplicate exports.
        The identifier is used to determine when the same function or class is
        exported many times. This can yield false positives.
        Args:
            function_or_class: The function or class to compute an identifier
                for.
        Returns:
            The identifier. Note that different functions or classes can give
                rise to same identifier. However, the same function should
                hopefully always give rise to the same identifier. TODO(rkn):
                verify if this is actually the case. Note that if the
                identifier is incorrect in any way, then we may give warnings
                unnecessarily or fail to give warnings, but the application's
                behavior won't change.
        """
    import io
    string_file = io.StringIO()
    dis.dis(function_or_class, file=string_file, depth=2)
    collision_identifier = function_or_class.__name__ + ':' + string_file.getvalue()
    return hashlib.sha1(collision_identifier.encode('utf-8')).digest()