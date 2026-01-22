from __future__ import annotations
import datetime
import os
import platform
import sys
import typing as t
from ...config import (
from ...io import (
from ...util import (
from ...util_common import (
from ...docker_util import (
from ...constants import (
from ...ci import (
from ...timeout import (
def command_env(args: EnvConfig) -> None:
    """Entry point for the `env` command."""
    show_dump_env(args)
    list_files_env(args)
    set_timeout(args)