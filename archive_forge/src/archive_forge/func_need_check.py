from abc import ABCMeta
import glob
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import warnings
from ray.util.annotations import PublicAPI, DeveloperAPI
from ray.tune.utils.util import _atomic_save, _load_newest_checkpoint
@classmethod
def need_check(mcs, cls: type, name: str, bases: Tuple[type], attrs: Dict[str, Any]) -> bool:
    return attrs.get('IS_CALLBACK_CONTAINER', False)