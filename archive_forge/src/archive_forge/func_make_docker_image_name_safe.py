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
def make_docker_image_name_safe(name: str) -> str:
    """Make a docker image name safe for use in artifacts."""
    safe_chars = RE_DOCKER_IMAGE_NAME_CHARS.sub('__', name.lower())
    deduped = RE_DOCKER_IMAGE_NAME_SEPARATOR_REPEAT.sub('__', safe_chars)
    trimmed_start = RE_DOCKER_IMAGE_NAME_SEPARATOR_START.sub('', deduped)
    trimmed = RE_DOCKER_IMAGE_NAME_SEPARATOR_END.sub('', trimmed_start)
    return trimmed if trimmed else 'image'