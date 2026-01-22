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
def merge_into(self, other: t.Any) -> t.Any:
    """
        Merge with another earlier LazyConfigValue or an earlier container.
        This is useful when having global system-wide configuration files.

        Self is expected to have higher precedence.

        Parameters
        ----------
        other : LazyConfigValue or container

        Returns
        -------
        LazyConfigValue
            if ``other`` is also lazy, a reified container otherwise.
        """
    if isinstance(other, LazyConfigValue):
        other._extend.extend(self._extend)
        self._extend = other._extend
        self._prepend.extend(other._prepend)
        other._inserts.extend(self._inserts)
        self._inserts = other._inserts
        if self._update:
            other.update(self._update)
            self._update = other._update
        return self
    else:
        return self.get_value(other)