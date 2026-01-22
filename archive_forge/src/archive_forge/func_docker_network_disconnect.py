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
def docker_network_disconnect(args: CommonConfig, container_id: str, network: str) -> None:
    """Disconnect the specified docker container from the given network."""
    docker_command(args, ['network', 'disconnect', network, container_id], capture=True)