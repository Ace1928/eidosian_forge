import functools
import logging
import os
import platform
import subprocess
from typing import Dict, List, Optional, Union
from langsmith.utils import get_docker_compose_command
from langsmith.env._git import exec_git
def get_runtime_and_metrics() -> dict:
    """Get the runtime information as well as metrics."""
    return {**get_runtime_environment(), **get_system_metrics()}