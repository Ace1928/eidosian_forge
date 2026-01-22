from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
@property
def current_namespace(self) -> t.Any:
    """The current namespace."""
    return self.namespaces[-1]