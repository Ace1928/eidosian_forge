import os
from pathlib import Path
import sys
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import iniconfig
from .exceptions import UsageError
from _pytest.outcomes import fail
from _pytest.pathlib import absolutepath
from _pytest.pathlib import commonpath
from _pytest.pathlib import safe_exists
def get_common_ancestor(invocation_dir: Path, paths: Iterable[Path]) -> Path:
    common_ancestor: Optional[Path] = None
    for path in paths:
        if not path.exists():
            continue
        if common_ancestor is None:
            common_ancestor = path
        elif common_ancestor in path.parents or path == common_ancestor:
            continue
        elif path in common_ancestor.parents:
            common_ancestor = path
        else:
            shared = commonpath(path, common_ancestor)
            if shared is not None:
                common_ancestor = shared
    if common_ancestor is None:
        common_ancestor = invocation_dir
    elif common_ancestor.is_file():
        common_ancestor = common_ancestor.parent
    return common_ancestor