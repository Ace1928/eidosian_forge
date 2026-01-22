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
def collect_uninstall(packages: list[str], ignore_errors: bool=False) -> list[PipUninstall]:
    """Return the details necessary for the specified pip uninstall."""
    uninstall = PipUninstall(packages=packages, ignore_errors=ignore_errors)
    return [uninstall]