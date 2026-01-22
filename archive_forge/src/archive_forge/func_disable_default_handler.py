import functools
import logging
import os
import sys
import threading
from logging import (
from logging import captureWarnings as _captureWarnings
from typing import Optional
import huggingface_hub.utils as hf_hub_utils
from tqdm import auto as tqdm_lib
def disable_default_handler() -> None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()
    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)