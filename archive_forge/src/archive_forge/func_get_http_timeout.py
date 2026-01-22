import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_http_timeout(default: int=20, env: Optional[Env]=None) -> int:
    if env is None:
        env = os.environ
    return int(env.get(HTTP_TIMEOUT, default))