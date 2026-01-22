from __future__ import annotations
import abc
import datetime
import os
import re
import tempfile
import time
import typing as t
from ....encoding import (
from ....io import (
from ....util import (
from ....util_common import (
from ....target import (
from ....config import (
from ....ci import (
from ....data import (
from ....docker_util import (
@cache
def get_environment_plugins() -> dict[str, t.Type[CloudEnvironment]]:
    """Return a dictionary of the available cloud environment plugins."""
    return get_cloud_plugins()[1]