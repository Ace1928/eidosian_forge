from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
def get_options_used(self) -> tuple[str, ...]:
    """Return a tuple of the command line options used."""
    fields: tuple[dataclasses.Field, ...] = dataclasses.fields(self)
    options = tuple(sorted((get_option_name(field.name) for field in fields if getattr(self, field.name))))
    return options