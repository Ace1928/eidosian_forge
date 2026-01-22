import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def should_save_code() -> bool:
    save_code = _env_as_bool(SAVE_CODE, default='False')
    code_disabled = _env_as_bool(DISABLE_CODE, default='False')
    return save_code and (not code_disabled)