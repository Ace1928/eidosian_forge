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
def sync_up_if_needed(self, local_dir: str, remote_dir: str, exclude: Optional[List]=None) -> bool:
    """Syncs up if time since last sync up is greater than sync_period.

        Args:
            local_dir: Local directory to sync from.
            remote_dir: Remote directory to sync up to. This is an URI
                (``protocol://remote/path``).
            exclude: Pattern of files to exclude, e.g.
                ``["*/checkpoint_*]`` to exclude trial checkpoints.
        """
    now = time.time()
    if now - self.last_sync_up_time >= self.sync_period:
        result = self.sync_up(local_dir=local_dir, remote_dir=remote_dir, exclude=exclude)
        self.last_sync_up_time = now
        return result