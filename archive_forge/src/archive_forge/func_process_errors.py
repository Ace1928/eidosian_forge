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
def process_errors(self, errors: list[SanityMessage], paths: list[str]) -> list[SanityMessage]:
    """Return the given errors filtered for ignores and with any settings related errors included."""
    errors = self.filter_messages(errors)
    errors.extend(self.get_errors(paths))
    errors = sorted(set(errors))
    return errors