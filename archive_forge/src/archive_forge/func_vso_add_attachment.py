from __future__ import annotations
import os
import tempfile
import uuid
import typing as t
import urllib.parse
from ..encoding import (
from ..config import (
from ..git import (
from ..http import (
from ..util import (
from . import (
def vso_add_attachment(file_type: str, file_name: str, path: str) -> None:
    """Upload and attach a file to the current timeline record."""
    vso('task.addattachment', dict(type=file_type, name=file_name), path)