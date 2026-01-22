import contextlib
import enum
import functools
import logging
import os
import pathlib
import subprocess
import threading
from typing import (
import requests
from urllib3.util import Retry
from langsmith import schemas as ls_schemas
def tracing_is_enabled() -> bool:
    """Return True if tracing is enabled."""
    var_result = get_env_var('TRACING_V2', default=get_env_var('TRACING', default=''))
    return var_result == 'true'