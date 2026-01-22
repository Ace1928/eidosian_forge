import os.path
import re
from contextlib import suppress
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Union
from .config import Config, get_xdg_config_home_path
def match_pattern(path: bytes, pattern: bytes, ignorecase: bool=False) -> bool:
    """Match a gitignore-style pattern against a path.

    Args:
      path: Path to match
      pattern: Pattern to match
      ignorecase: Whether to do case-sensitive matching
    Returns:
      bool indicating whether the pattern matched
    """
    return Pattern(pattern, ignorecase).match(path)