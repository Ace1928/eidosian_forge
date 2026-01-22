import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_use_v1_artifacts(env: Optional[Env]=None) -> bool:
    if env is None:
        env = os.environ
    val = bool(env.get(USE_V1_ARTIFACTS, False))
    return val