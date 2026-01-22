import os
from pathlib import Path
from typing import List, Optional
def xdg_state_home() -> Path:
    """Return a Path corresponding to XDG_STATE_HOME."""
    return _path_from_env('XDG_STATE_HOME', Path.home() / '.local' / 'state')