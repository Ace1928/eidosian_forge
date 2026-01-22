import os
from pathlib import Path
from typing import List, Optional
def xdg_data_dirs() -> List[Path]:
    """Return a list of Paths corresponding to XDG_DATA_DIRS."""
    return _paths_from_env('XDG_DATA_DIRS', [Path(path) for path in '/usr/local/share/:/usr/share/'.split(':')])