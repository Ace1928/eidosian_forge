import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_launch_trace_id(env: Optional[Env]=None) -> Optional[str]:
    if env is None:
        env = os.environ
    val = env.get(LAUNCH_TRACE_ID, None)
    return val