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
def sync_down(self, remote_dir: str, local_dir: str, exclude: Optional[List]=None) -> bool:
    if self._should_continue_existing_sync():
        logger.warning(f'Last sync still in progress, skipping sync down of {remote_dir} to {local_dir}')
        return False
    sync_down_cmd = self._sync_down_command(uri=remote_dir, local_path=local_dir)
    self._launch_sync_process(sync_down_cmd)
    return True