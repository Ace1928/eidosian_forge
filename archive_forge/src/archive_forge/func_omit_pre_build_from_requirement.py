from __future__ import annotations
import abc
import glob
import hashlib
import json
import os
import pathlib
import re
import collections
import collections.abc as c
import typing as t
from ...constants import (
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...executor import (
from ...python_requirements import (
from ...config import (
from ...test import (
from ...data import (
from ...content_config import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...venv import (
def omit_pre_build_from_requirement(path: str, requirements: str) -> tuple[str, str]:
    """Return the given requirements with pre-build instructions omitted."""
    lines = requirements.splitlines(keepends=True)
    lines = [line for line in lines if not line.startswith('# pre-build ')]
    return (path, ''.join(lines))