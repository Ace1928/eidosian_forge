from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
@abc.abstractmethod
def get_stateless_parsers(self) -> dict[str, Parser]:
    """Return a dictionary of type names and type parsers."""