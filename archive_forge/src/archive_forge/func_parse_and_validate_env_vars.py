import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import yaml
def parse_and_validate_env_vars(env_vars: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Parses and validates a user-provided 'env_vars' option.

    This is validated to verify that all keys and vals are strings.

    If an empty dictionary is passed, we return `None` for consistency.

    Args:
        env_vars: A dictionary of environment variables to set in the
            runtime environment.

    Returns:
        The validated env_vars dictionary, or None if it was empty.

    Raises:
        TypeError: If the env_vars is not a dictionary of strings. The error message
            will include the type of the invalid value.
    """
    assert env_vars is not None
    if len(env_vars) == 0:
        return None
    if not isinstance(env_vars, dict):
        raise TypeError(f"runtime_env['env_vars'] must be of type Dict[str, str], got {type(env_vars)}")
    for key, val in env_vars.items():
        if not isinstance(key, str):
            raise TypeError(f"runtime_env['env_vars'] must be of type Dict[str, str], but the key {key} is of type {type(key)}")
        if not isinstance(val, str):
            raise TypeError(f"runtime_env['env_vars'] must be of type Dict[str, str], but the value {val} is of type {type(val)}")
    return env_vars