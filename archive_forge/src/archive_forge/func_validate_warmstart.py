import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
@PublicAPI()
def validate_warmstart(parameter_names: List[str], points_to_evaluate: List[Union[List, Dict]], evaluated_rewards: List, validate_point_name_lengths: bool=True):
    """Generic validation of a Searcher's warm start functionality.
    Raises exceptions in case of type and length mismatches between
    parameters.

    If ``validate_point_name_lengths`` is False, the equality of lengths
    between ``points_to_evaluate`` and ``parameter_names`` will not be
    validated.
    """
    if points_to_evaluate:
        if not isinstance(points_to_evaluate, list):
            raise TypeError('points_to_evaluate expected to be a list, got {}.'.format(type(points_to_evaluate)))
        for point in points_to_evaluate:
            if not isinstance(point, (dict, list)):
                raise TypeError(f'points_to_evaluate expected to include list or dict, got {point}.')
            if validate_point_name_lengths and (not len(point) == len(parameter_names)):
                raise ValueError('Dim of point {}'.format(point) + ' and parameter_names {}'.format(parameter_names) + ' do not match.')
    if points_to_evaluate and evaluated_rewards:
        if not isinstance(evaluated_rewards, list):
            raise TypeError('evaluated_rewards expected to be a list, got {}.'.format(type(evaluated_rewards)))
        if not len(evaluated_rewards) == len(points_to_evaluate):
            raise ValueError('Dim of evaluated_rewards {}'.format(evaluated_rewards) + ' and points_to_evaluate {}'.format(points_to_evaluate) + ' do not match.')