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
def usable_pip_file(path: t.Optional[str]) -> bool:
    """Return True if the specified pip file is usable, otherwise False."""
    return bool(path) and os.path.exists(path) and bool(os.path.getsize(path))