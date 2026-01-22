import colorsys
import contextlib
import dataclasses
import functools
import gzip
import importlib
import importlib.util
import itertools
import json
import logging
import math
import numbers
import os
import platform
import queue
import random
import re
import secrets
import shlex
import socket
import string
import sys
import tarfile
import tempfile
import threading
import time
import types
import urllib
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from importlib import import_module
from sys import getsizeof
from types import ModuleType
from typing import (
import requests
import yaml
import wandb
import wandb.env
from wandb.errors import AuthenticationError, CommError, UsageError, term
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, runid
from wandb.sdk.lib.json_util import dump, dumps
from wandb.sdk.lib.paths import FilePathStr, StrPath
def check_retry_conflict(e: Any) -> Optional[bool]:
    """Check if the exception is a conflict type so it can be retried.

    Returns:
        True - Should retry this operation
        False - Should not retry this operation
        None - No decision, let someone else decide
    """
    if hasattr(e, 'exception'):
        e = e.exception
    if isinstance(e, requests.HTTPError) and e.response is not None:
        if e.response.status_code == 409:
            return True
    return None