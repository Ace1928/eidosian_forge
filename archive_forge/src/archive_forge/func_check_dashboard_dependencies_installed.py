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
def check_dashboard_dependencies_installed() -> bool:
    """Returns True if Ray Dashboard dependencies are installed.

    Checks to see if we should start the dashboard agent or not based on the
    Ray installation version the user has installed (ray vs. ray[default]).
    Unfortunately there doesn't seem to be a cleaner way to detect this other
    than just blindly importing the relevant packages.

    """
    try:
        import ray.dashboard.optional_deps
        return True
    except ImportError:
        return False