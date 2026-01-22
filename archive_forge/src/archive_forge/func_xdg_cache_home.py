import os
from pathlib import Path
from typing import List, Optional
def xdg_cache_home() -> Path:
    """Return a Path corresponding to XDG_CACHE_HOME."""
    return _path_from_env('XDG_CACHE_HOME', Path.home() / '.cache')