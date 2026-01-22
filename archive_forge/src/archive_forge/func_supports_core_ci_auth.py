from __future__ import annotations
import abc
import base64
import json
import os
import tempfile
import typing as t
from ..encoding import (
from ..io import (
from ..config import (
from ..util import (
@abc.abstractmethod
def supports_core_ci_auth(self) -> bool:
    """Return True if Ansible Core CI is supported."""