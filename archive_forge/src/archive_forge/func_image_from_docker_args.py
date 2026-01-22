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
def image_from_docker_args(args: List[str]) -> Optional[str]:
    """Scan docker run args and attempt to find the most likely docker image argument.

    It excludes any arguments that start with a dash, and the argument after it if it
    isn't a boolean switch. This can be improved, we currently fallback gracefully when
    this fails.
    """
    bool_args = ['-t', '--tty', '--rm', '--privileged', '--oom-kill-disable', '--no-healthcheck', '-i', '--interactive', '--init', '--help', '--detach', '-d', '--sig-proxy', '-it', '-itd']
    last_flag = -2
    last_arg = ''
    possible_images = []
    if len(args) > 0 and args[0] == 'run':
        args.pop(0)
    for i, arg in enumerate(args):
        if arg.startswith('-'):
            last_flag = i
            last_arg = arg
        elif '@sha256:' in arg:
            possible_images.append(arg)
        elif docker_image_regex(arg):
            if last_flag == i - 2:
                possible_images.append(arg)
            elif '=' in last_arg:
                possible_images.append(arg)
            elif last_arg in bool_args and last_flag == i - 1:
                possible_images.append(arg)
    most_likely = None
    for img in possible_images:
        if ':' in img or '@' in img or '/' in img:
            most_likely = img
            break
    if most_likely is None and len(possible_images) > 0:
        most_likely = possible_images[0]
    return most_likely