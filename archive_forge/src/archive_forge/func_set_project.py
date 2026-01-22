import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def set_project(value: str, env: Optional[Env]=None) -> None:
    if env is None:
        env = os.environ
    env[PROJECT] = value or 'uncategorized'