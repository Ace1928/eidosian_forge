from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
def get_parsers(self, state: ParserState) -> dict[str, Parser]:
    """Return a dictionary of type names and type parsers."""
    return self.get_stateless_parsers()