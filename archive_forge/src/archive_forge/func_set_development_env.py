from __future__ import annotations
import traceback
from lazyops.utils.logs import logger
from fastapi.exceptions import HTTPException
from typing import Any, Optional, Union
def set_development_env(is_development: bool):
    """
    Sets the development environment
    """
    global _is_development_env
    _is_development_env = is_development