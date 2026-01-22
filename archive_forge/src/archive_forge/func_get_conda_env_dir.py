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
def get_conda_env_dir(env_name):
    """Find and validate the conda directory for a given conda environment.

    For example, given the environment name `tf1`, this function checks
    the existence of the corresponding conda directory, e.g.
    `/Users/scaly/anaconda3/envs/tf1`, and returns it.
    """
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix is None:
        conda_exe = os.environ.get('CONDA_EXE')
        if conda_exe is None:
            raise ValueError('Cannot find environment variables set by conda. Please verify conda is installed.')
        conda_prefix = str(Path(conda_exe).parent.parent)
    if os.environ.get('CONDA_DEFAULT_ENV') == 'base':
        if env_name == 'base':
            env_dir = conda_prefix
        else:
            env_dir = os.path.join(conda_prefix, 'envs', env_name)
    else:
        conda_envs_dir = os.path.split(conda_prefix)[0]
        env_dir = os.path.join(conda_envs_dir, env_name)
    if not os.path.isdir(env_dir):
        raise ValueError('conda env ' + env_name + ' not found in conda envs directory. Run `conda env list` to ' + 'verify the name is correct.')
    return env_dir