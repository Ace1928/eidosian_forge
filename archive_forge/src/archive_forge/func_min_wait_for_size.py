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
@classmethod
def min_wait_for_size(cls, size: int) -> float:
    if size < 10 * cls.unit_dict['MB']:
        return 60
    elif size < 100 * cls.unit_dict['MB']:
        return 5 * 60
    elif size < cls.unit_dict['GB']:
        return 10 * 60
    else:
        return 20 * 60