import functools
import logging
import os
import platform
import subprocess
from typing import Dict, List, Optional, Union
from langsmith.utils import get_docker_compose_command
from langsmith.env._git import exec_git
@functools.lru_cache(maxsize=1)
def get_langchain_environment() -> Optional[str]:
    try:
        import langchain
        return langchain.__version__
    except:
        return None