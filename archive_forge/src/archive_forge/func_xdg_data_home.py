import os
from pathlib import Path
from typing import List, Optional
def xdg_data_home() -> Path:
    """Return a Path corresponding to XDG_DATA_HOME."""
    return _path_from_env('XDG_DATA_HOME', Path.home() / '.local' / 'share')