import os.path
import re
from contextlib import suppress
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Union
from .config import Config, get_xdg_config_home_path
def default_user_ignore_filter_path(config: Config) -> str:
    """Return default user ignore filter path.

    Args:
      config: A Config object
    Returns:
      Path to a global ignore file
    """
    try:
        value = config.get((b'core',), b'excludesFile')
        assert isinstance(value, bytes)
        return value.decode(encoding='utf-8')
    except KeyError:
        pass
    return get_xdg_config_home_path('git', 'ignore')