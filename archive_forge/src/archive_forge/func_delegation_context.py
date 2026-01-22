from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import tempfile
import typing as t
from .constants import (
from .locale_util import (
from .io import (
from .config import (
from .util import (
from .util_common import (
from .ansible_util import (
from .containers import (
from .data import (
from .payload import (
from .ci import (
from .host_configs import (
from .connections import (
from .provisioning import (
from .content_config import (
@contextlib.contextmanager
def delegation_context(args: EnvironmentConfig, host_state: HostState) -> c.Iterator[None]:
    """Context manager for serialized host state during delegation."""
    make_dirs(ResultType.TMP.path)
    python = host_state.controller_profile.python
    del python
    with tempfile.TemporaryDirectory(prefix='host-', dir=ResultType.TMP.path) as host_dir:
        args.host_settings.serialize(os.path.join(host_dir, 'settings.dat'))
        host_state.serialize(os.path.join(host_dir, 'state.dat'))
        serialize_content_config(args, os.path.join(host_dir, 'config.dat'))
        args.host_path = os.path.join(ResultType.TMP.relative_path, os.path.basename(host_dir))
        try:
            yield
        finally:
            args.host_path = None