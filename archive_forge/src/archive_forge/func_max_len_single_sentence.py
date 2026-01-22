import copy
import json
import os
import re
import warnings
from collections import UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
@max_len_single_sentence.setter
def max_len_single_sentence(self, value) -> int:
    if value == self.model_max_length - self.num_special_tokens_to_add(pair=False) and self.verbose:
        if not self.deprecation_warnings.get('max_len_single_sentence', False):
            logger.warning("Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up.")
        self.deprecation_warnings['max_len_single_sentence'] = True
    else:
        raise ValueError("Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up.")