import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def update_envs(env_vars: Dict[str, str]):
    """
    When updating the environment variable, if there is ${X},
    it will be replaced with the current environment variable.
    """
    if not env_vars:
        return
    for key, value in env_vars.items():
        expanded = os.path.expandvars(value)
        result = re.sub('\\$\\{[A-Z0-9_]+\\}', '', expanded)
        os.environ[key] = result