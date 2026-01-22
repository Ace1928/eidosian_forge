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
def hide_bars(self) -> None:
    """Temporarily hide visible bars to avoid conflict with other log messages."""
    with self.lock:
        if not self.in_hidden_state:
            self.in_hidden_state = True
            self.num_hides += 1
            for group in self.bar_groups.values():
                group.hide_bars()