import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_ignore(default: Optional[List[str]]=None, env: Optional[Env]=None) -> Optional[List[str]]:
    if env is None:
        env = os.environ
    ignore = env.get(IGNORE)
    if ignore is not None:
        return ignore.split(',')
    else:
        return default