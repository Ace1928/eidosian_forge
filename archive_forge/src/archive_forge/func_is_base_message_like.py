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
def is_base_message_like(obj: object) -> bool:
    """Check if the given object is similar to BaseMessage.

    Args:
        obj (object): The object to check.

    Returns:
        bool: True if the object is similar to BaseMessage, False otherwise.
    """
    return all([isinstance(getattr(obj, 'content', None), str), isinstance(getattr(obj, 'additional_kwargs', None), dict), hasattr(obj, 'type') and isinstance(getattr(obj, 'type'), str)])