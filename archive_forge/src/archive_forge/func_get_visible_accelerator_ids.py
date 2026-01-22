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
def get_visible_accelerator_ids() -> Mapping[str, Optional[List[str]]]:
    """Get the mapping from accelerator resource name
    to the visible ids."""
    from ray._private.accelerators import get_all_accelerator_resource_names, get_accelerator_manager_for_resource
    return {accelerator_resource_name: get_accelerator_manager_for_resource(accelerator_resource_name).get_current_process_visible_accelerator_ids() for accelerator_resource_name in get_all_accelerator_resource_names()}