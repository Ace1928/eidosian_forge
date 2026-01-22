from __future__ import annotations
import ast
import builtins
import sys
import typing
from ast import (
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast, overload
def get_config_keywords(self) -> list[keyword]:
    if self.parent and isinstance(self.parent.node, ClassDef):
        overrides = self.parent.configuration_overrides.copy()
    else:
        overrides = {}
    overrides.update(self.configuration_overrides)
    return [keyword(key, value) for key, value in overrides.items()]