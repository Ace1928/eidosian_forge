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
def make_pip_command_hashable(command: PipCommand) -> tuple[str, dict[str, t.Any]]:
    """Return a serialized version of the given pip command that is suitable for hashing."""
    if isinstance(command, PipInstall):
        command = PipInstall(requirements=[omit_pre_build_from_requirement(*req) for req in command.requirements], constraints=list(command.constraints), packages=list(command.packages))
    return command.serialize()