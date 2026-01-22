import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from dockerpycreds.utils import find_executable  # type: ignore
from wandb.docker import auth, www_authenticate
from wandb.errors import Error
def should_add_load_argument(platform: Optional[str]) -> bool:
    if platform is None or (platform and ',' not in platform):
        return True
    return False