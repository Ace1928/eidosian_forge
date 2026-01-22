from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import re
import socket
import time
import urllib.parse
import typing as t
from .util import (
from .util_common import (
from .config import (
from .thread import (
from .cgroup import (
@mutex
def get_docker_info(args: CommonConfig) -> DockerInfo:
    """Return info for the current container runtime. The results are cached."""
    try:
        return get_docker_info.info
    except AttributeError:
        pass
    info = DockerInfo.init(args)
    display.info(info.description, verbosity=1)
    get_docker_info.info = info
    return info