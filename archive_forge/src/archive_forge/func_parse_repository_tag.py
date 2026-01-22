import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from dockerpycreds.utils import find_executable  # type: ignore
from wandb.docker import auth, www_authenticate
from wandb.errors import Error
def parse_repository_tag(repo_name: str) -> Tuple[str, Optional[str]]:
    parts = repo_name.rsplit('@', 1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    parts = repo_name.rsplit(':', 1)
    if len(parts) == 2 and '/' not in parts[1]:
        return (parts[0], parts[1])
    return (repo_name, None)