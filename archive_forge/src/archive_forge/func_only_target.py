from __future__ import annotations
import dataclasses
import enum
import os
import sys
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .data import (
from .host_configs import (
def only_target(self, target_type: t.Type[THostConfig]) -> THostConfig:
    """
        Return the host configuration for the target.
        Requires that there is exactly one target of the specified type.
        """
    targets = list(self.targets)
    if len(targets) != 1:
        raise Exception('There must be exactly one target.')
    target = targets.pop()
    if not isinstance(target, target_type):
        raise Exception(f'Target is {type(target_type)} instead of {target_type}.')
    return target