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
def get_entrypoint_name():
    """Get the entrypoint of the current script."""
    prefix = ''
    try:
        curr = psutil.Process()
        if hasattr(sys, 'ps1'):
            prefix = '(interactive_shell) '
        return prefix + list2cmdline(curr.cmdline())
    except Exception:
        return 'unknown'