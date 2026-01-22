import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from asyncio import create_task, get_running_loop
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import Protocol, parse_uri
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.utils import check_output_cmd
from ray._private.utils import get_directory_size_bytes, try_to_create_directory
import ray
@staticmethod
def get_requirements_file(target_dir: str, pip_list: Optional[List[str]]) -> str:
    """Returns the path to the requirements file to use for this runtime env.

        If pip_list is not None, we will check if the internal pip filename is in any of
        the entries of pip_list. If so, we will append numbers to the end of the
        filename until we find one that doesn't conflict. This prevents infinite
        recursion if the user specifies the internal pip filename in their pip list.

        Args:
            target_dir: The directory to store the requirements file in.
            pip_list: A list of pip requirements specified by the user.

        Returns:
            The path to the requirements file to use for this runtime env.
        """

    def filename_in_pip_list(filename: str) -> bool:
        for pip_entry in pip_list:
            if filename in pip_entry:
                return True
        return False
    filename = INTERNAL_PIP_FILENAME
    if pip_list is not None:
        i = 1
        while filename_in_pip_list(filename) and i < MAX_INTERNAL_PIP_FILENAME_TRIES:
            filename = f'{INTERNAL_PIP_FILENAME}.{i}'
            i += 1
        if i == MAX_INTERNAL_PIP_FILENAME_TRIES:
            raise RuntimeError('Could not find a valid filename for the internal pip requirements file. Please specify a different pip list in your runtime env.')
    return os.path.join(target_dir, filename)