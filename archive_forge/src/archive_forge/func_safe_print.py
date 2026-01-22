import builtins
import copy
import json
import logging
import os
import sys
import threading
import uuid
from typing import Any, Dict, Iterable, Optional
import colorama
import ray
from ray._private.ray_constants import env_bool
from ray.util.debug import log_once
def safe_print(*args, **kwargs):
    """Use this as an alternative to `print` that will not corrupt tqdm output.

    By default, the builtin print will be patched to this function when tqdm_ray is
    used. To disable this, set RAY_TQDM_PATCH_PRINT=0.
    """
    if kwargs.get('file') not in [sys.stdout, sys.stderr, None]:
        return _print(*args, **kwargs)
    try:
        instance().hide_bars()
        _print(*args, **kwargs)
    finally:
        instance().unhide_bars()