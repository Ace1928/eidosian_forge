from __future__ import annotations
import collections
import contextlib
import json
import os
import tarfile
import typing as t
from . import (
from ...io import (
from ...test import (
from ...target import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...config import (
from ...ci import (
from ...data import (
from ...host_configs import (
from ...git import (
from ...provider.source import (
def git_callback(payload_config: PayloadConfig) -> None:
    """Include the previous plugin content archive in the payload."""
    files = payload_config.files
    files.append((path, os.path.relpath(path, data_context().content.root)))