import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import yaml
Parses and validates a user-provided 'env_vars' option.

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
    