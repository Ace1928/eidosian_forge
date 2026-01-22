from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
def is_cryptography_available(python: str) -> bool:
    """Return True if cryptography is available for the given python."""
    try:
        raw_command([python, '-c', 'import cryptography'], capture=True)
    except SubprocessError:
        return False
    return True