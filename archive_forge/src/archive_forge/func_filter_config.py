import functools
import logging
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy
import wasabi.tables
from .. import util
from ..errors import Errors
from ..pipeline import MultiLabel_TextCategorizer, TextCategorizer
from ..training import Corpus
from ._util import Arg, Opt, app, import_code, setup_gpu
def filter_config(config: Dict[str, Any], keys: List[str], full_key: str) -> Dict[str, Any]:
    """Filters provided config dictionary so that only the specified keys path remains.
        config (Dict[str, Any]): Configuration dictionary.
        keys (List[Any]): Path to value to set.
        full_key (str): Full user-specified key.
        RETURNS (Dict[str, Any]): Filtered dictionary.
        """
    if keys[0] not in config:
        wasabi.msg.fail(title=f'Failed to look up `{full_key}` in config: sub-key {[keys[0]]} not found.', text=f'Make sure you specified {[keys[0]]} correctly. The following sub-keys are available instead: {list(config.keys())}', exits=1)
    return {keys[0]: filter_config(config[keys[0]], keys[1:], full_key) if len(keys) > 1 else config[keys[0]]}