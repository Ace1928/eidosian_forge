import abc
import logging
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.train.constants import _DEPRECATED_VALUE
from ray.util import log_once
from ray.util.annotations import PublicAPI
from ray.widgets import Template
def sync_down_if_needed(self, remote_dir: str, local_dir: str, exclude: Optional[List]=None):
    """Syncs down if time since last sync down is greater than sync_period.

        Args:
            remote_dir: Remote directory to sync down from. This is an URI
                (``protocol://remote/path``).
            local_dir: Local directory to sync to.
            exclude: Pattern of files to exclude, e.g.
                ``["*/checkpoint_*]`` to exclude trial checkpoints.
        """
    now = time.time()
    if now - self.last_sync_down_time >= self.sync_period:
        result = self.sync_down(remote_dir=remote_dir, local_dir=local_dir, exclude=exclude)
        self.last_sync_down_time = now
        return result