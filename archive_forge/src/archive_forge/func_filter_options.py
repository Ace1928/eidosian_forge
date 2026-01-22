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
def filter_options(args: EnvironmentConfig, argv: list[str], exclude: list[str], require: list[str]) -> c.Iterable[str]:
    """Return an iterable that filters out unwanted CLI options and injects new ones as requested."""
    replace: list[tuple[str, int, t.Optional[t.Union[bool, str, list[str]]]]] = [('--truncate', 1, str(args.truncate)), ('--color', 1, 'yes' if args.color else 'no'), ('--redact', 0, False), ('--no-redact', 0, not args.redact), ('--host-path', 1, args.host_path)]
    if isinstance(args, TestConfig):
        replace.extend([('--changed', 0, False), ('--tracked', 0, False), ('--untracked', 0, False), ('--ignore-committed', 0, False), ('--ignore-staged', 0, False), ('--ignore-unstaged', 0, False), ('--changed-from', 1, False), ('--changed-path', 1, False), ('--metadata', 1, args.metadata_path), ('--exclude', 1, exclude), ('--require', 1, require), ('--base-branch', 1, False)])
    pass_through_args: list[str] = []
    for arg in filter_args(argv, {option: count for option, count, replacement in replace}):
        if arg == '--' or pass_through_args:
            pass_through_args.append(arg)
            continue
        yield arg
    for option, _count, replacement in replace:
        if not replacement:
            continue
        if isinstance(replacement, bool):
            yield option
        elif isinstance(replacement, str):
            yield from [option, replacement]
        elif isinstance(replacement, list):
            for item in replacement:
                yield from [option, item]
    yield from args.delegate_args
    yield from pass_through_args