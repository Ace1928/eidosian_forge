import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import yaml
def parse_and_validate_excludes(excludes: List[str]) -> List[str]:
    """Parses and validates a user-provided 'excludes' option.

    This is validated to verify that it is of type List[str].

    If an empty list is passed, we return `None` for consistency.
    """
    assert excludes is not None
    if isinstance(excludes, list) and len(excludes) == 0:
        return None
    if isinstance(excludes, list) and all((isinstance(path, str) for path in excludes)):
        return excludes
    else:
        raise TypeError(f"runtime_env['excludes'] must be of type List[str], got {type(excludes)}")