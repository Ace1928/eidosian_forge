from __future__ import annotations
import argparse
import enum
import functools
import typing as t
from ..constants import (
from ..util import (
from ..completion import (
from ..cli.argparsing import (
from ..cli.argparsing.actions import (
from ..cli.actions import (
from ..cli.compat import (
from ..config import (
from .completers import (
from .converters import (
from .epilog import (
from ..ci import (
def complete_network_platform(prefix: str, parsed_args: argparse.Namespace, **_) -> list[str]:
    """Return a list of supported network platforms matching the given prefix, excluding platforms already parsed from the command line."""
    images = sorted(filter_completion(network_completion()))
    return [i for i in images if i.startswith(prefix) and (not parsed_args.platform or i not in parsed_args.platform)]