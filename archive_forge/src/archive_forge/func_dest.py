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
def dest(self) -> str:
    """The name of the attribute where the value should be stored."""
    return self._dest