from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
@_requires_ips
def is_local_ip(ip: str) -> bool:
    """does `ip` point to this machine?"""
    return ip in LOCAL_IPS