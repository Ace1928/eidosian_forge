import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_agent_max_initial_failures(default: Optional[int]=None, env: Optional[Env]=None) -> Optional[int]:
    if env is None:
        env = os.environ
    val = env.get(AGENT_MAX_INITIAL_FAILURES, default)
    try:
        val = int(val)
    except ValueError:
        val = default
    return val