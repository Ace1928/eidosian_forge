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
def pasre_pg_formatted_resources_to_original(pg_formatted_resources: Dict[str, float]) -> Dict[str, float]:
    original_resources = {}
    for key, value in pg_formatted_resources.items():
        result = PLACEMENT_GROUP_WILDCARD_RESOURCE_PATTERN.match(key)
        if result and len(result.groups()) == 2:
            original_resources[result.group(1)] = value
            continue
        result = PLACEMENT_GROUP_INDEXED_BUNDLED_RESOURCE_PATTERN.match(key)
        if result and len(result.groups()) == 3:
            original_resources[result.group(1)] = value
            continue
        original_resources[key] = value
    return original_resources