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
def update_bar(self, state: ProgressBarState) -> None:
    """Update the state of a managed bar in this group."""
    bar = self.bars_by_uuid[state['uuid']]
    bar.update(state)