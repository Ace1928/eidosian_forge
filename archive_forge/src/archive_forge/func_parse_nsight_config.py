import os
import sys
import logging
import asyncio
import subprocess
import copy
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.utils import (
from ray.exceptions import RuntimeEnvSetupError
def parse_nsight_config(nsight_config: Dict[str, str]) -> List[str]:
    """
    Function to convert dictionary of nsight options into
    nsight command line

    The function returns:
    - List[str]: nsys profile cmd line split into list of str
    """
    nsight_cmd = ['nsys', 'profile']
    for option, option_val in nsight_config.items():
        if len(option) > 1:
            nsight_cmd.append(f'--{option}={option_val}')
        else:
            nsight_cmd += [f'-{option}', option_val]
    return nsight_cmd