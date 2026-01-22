import copy
import inspect
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from .dynamic_module_utils import custom_object_save
from .tokenization_utils_base import PreTrainedTokenizerBase
from .utils import (

        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoProcessor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoProcessor"`):
                The auto class to register this new feature extractor with.
        