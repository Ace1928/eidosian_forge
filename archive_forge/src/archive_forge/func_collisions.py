from __future__ import annotations
import argparse
import copy
import functools
import json
import os
import re
import sys
import typing as t
from logging import Logger
from traitlets.traitlets import Any, Container, Dict, HasTraits, List, TraitType, Undefined
from ..utils import cast_unicode, filefind, warnings
def collisions(self, other: Config) -> dict[str, t.Any]:
    """Check for collisions between two config objects.

        Returns a dict of the form {"Class": {"trait": "collision message"}}`,
        indicating which values have been ignored.

        An empty dict indicates no collisions.
        """
    collisions: dict[str, t.Any] = {}
    for section in self:
        if section not in other:
            continue
        mine = self[section]
        theirs = other[section]
        for key in mine:
            if key in theirs and mine[key] != theirs[key]:
                collisions.setdefault(section, {})
                collisions[section][key] = f'{mine[key]!r} ignored, using {theirs[key]!r}'
    return collisions