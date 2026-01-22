import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_error_reporting(default: Union[bool, str]=True, env: Optional[Env]=None) -> Union[bool, str]:
    if env is None:
        env = os.environ
    return env.get(ERROR_REPORTING, default)