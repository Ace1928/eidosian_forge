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
def slots_required(self):
    """Return the number of pos slots we need to accomodate bars in this group."""
    if not self.bars_by_uuid:
        return 0
    return 1 + max((bar.state['pos'] for bar in self.bars_by_uuid.values()))