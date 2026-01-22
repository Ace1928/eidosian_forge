import os
from pathlib import Path
from typing import List, Optional
def xdg_config_dirs() -> List[Path]:
    """Return a list of Paths corresponding to XDG_CONFIG_DIRS."""
    return _paths_from_env('XDG_CONFIG_DIRS', [Path('/etc/xdg')])