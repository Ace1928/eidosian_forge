import functools
import logging
import os
import platform
import subprocess
from typing import Dict, List, Optional, Union
from langsmith.utils import get_docker_compose_command
from langsmith.env._git import exec_git
def get_langchain_env_vars() -> dict:
    """Retrieve the langchain environment variables."""
    env_vars = {k: v for k, v in os.environ.items() if k.startswith('LANGCHAIN_')}
    for key in list(env_vars):
        if 'key' in key.lower():
            v = env_vars[key]
            env_vars[key] = v[:2] + '*' * (len(v) - 4) + v[-2:]
    return env_vars