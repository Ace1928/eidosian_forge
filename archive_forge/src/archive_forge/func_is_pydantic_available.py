import importlib.metadata
import platform
import sys
import warnings
from typing import Any, Dict
from .. import __version__, constants
def is_pydantic_available() -> bool:
    if not is_package_available('pydantic'):
        return False
    try:
        from pydantic import validator
    except ImportError:
        warnings.warn("Pydantic is installed but cannot be imported. Please check your installation. `huggingface_hub` will default to not using Pydantic. Error message: '{e}'")
        return False
    return True