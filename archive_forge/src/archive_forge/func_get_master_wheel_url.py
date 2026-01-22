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
def get_master_wheel_url(ray_commit: str=ray.__commit__, sys_platform: str=sys.platform, ray_version: str=ray.__version__, py_version: Tuple[int, int]=sys.version_info[:2]) -> str:
    """Return the URL for the wheel from a specific commit."""
    filename = get_wheel_filename(sys_platform=sys_platform, ray_version=ray_version, py_version=py_version)
    return f'https://s3-us-west-2.amazonaws.com/ray-wheels/master/{ray_commit}/{filename}'