from __future__ import annotations
import abc
import dataclasses
import json
import os
import re
import stat
import traceback
import uuid
import time
import typing as t
from .http import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .ci import (
from .data import (
def get_in_tree_key_pair_paths(self) -> t.Optional[tuple[str, str]]:
    """Return the ansible-test SSH key pair paths from the content tree."""
    temp_dir = ResultType.TMP.path
    key = os.path.join(temp_dir, self.KEY_NAME)
    pub = os.path.join(temp_dir, self.PUB_NAME)
    return (key, pub)