import abc
import fnmatch
import glob
import logging
import os
import queue
import time
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, MutableSet, Optional
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib.paths import LogicalPath
@property
def emitter(self) -> Optional['wd_api.EventEmitter']:
    try:
        return next(iter(self._file_observer.emitters))
    except StopIteration:
        return None