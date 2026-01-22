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
def parse_node_labels_json(labels_json: str, cli_logger, cf, command_arg='--labels') -> Dict[str, str]:
    try:
        labels = json.loads(labels_json)
        if not isinstance(labels, dict):
            raise ValueError('The format after deserialization is not a key-value pair map')
        for key, value in labels.items():
            if not isinstance(key, str):
                raise ValueError('The key is not string type.')
            if not isinstance(value, str):
                raise ValueError(f'The value of the "{key}" is not string type')
    except Exception as e:
        cli_logger.abort('`{}` is not a valid JSON string, detail error:{}Valid values look like this: `{}`', cf.bold(f'{command_arg}={labels_json}'), str(e), cf.bold(f"""{command_arg}='{{"gpu_type": "A100", "region": "us"}}'"""))
    return labels