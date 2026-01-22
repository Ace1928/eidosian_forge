import hashlib
import json
import logging
import os
import platform
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from filelock import FileLock
import ray
from ray._private.runtime_env.conda_utils import (
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import Protocol, parse_uri
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.validation import parse_and_validate_conda
from ray._private.utils import (
def inject_dependencies(conda_dict: Dict[Any, Any], py_version: str, pip_dependencies: Optional[List[str]]=None) -> Dict[Any, Any]:
    """Add Ray, Python and (optionally) extra pip dependencies to a conda dict.

    Args:
        conda_dict: A dict representing the JSON-serialized conda
            environment YAML file.  This dict will be modified and returned.
        py_version: A string representing a Python version to inject
            into the conda dependencies, e.g. "3.7.7"
        pip_dependencies (List[str]): A list of pip dependencies that
            will be prepended to the list of pip dependencies in
            the conda dict.  If the conda dict does not already have a "pip"
            field, one will be created.
    Returns:
        The modified dict.  (Note: the input argument conda_dict is modified
        and returned.)
    """
    if pip_dependencies is None:
        pip_dependencies = []
    if conda_dict.get('dependencies') is None:
        conda_dict['dependencies'] = []
    deps = conda_dict['dependencies']
    deps.append(f'python={py_version}')
    if 'pip' not in deps:
        deps.append('pip')
    found_pip_dict = False
    for dep in deps:
        if isinstance(dep, dict) and dep.get('pip') and isinstance(dep['pip'], list):
            dep['pip'] = pip_dependencies + dep['pip']
            found_pip_dict = True
            break
    if not found_pip_dict:
        deps.append({'pip': pip_dependencies})
    return conda_dict