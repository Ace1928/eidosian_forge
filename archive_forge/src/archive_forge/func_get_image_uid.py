import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from dockerpycreds.utils import find_executable  # type: ignore
from wandb.docker import auth, www_authenticate
from wandb.errors import Error
def get_image_uid(image_name: str) -> int:
    """Retrieve the image default uid through brute force."""
    image_uid = shell(['run', image_name, 'id', '-u'])
    return int(image_uid) if image_uid else -1