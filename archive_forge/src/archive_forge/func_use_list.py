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
def use_list(self) -> bool:
    """True if the destination is a list, otherwise False."""
    return False