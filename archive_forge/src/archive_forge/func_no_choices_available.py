from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
def no_choices_available(self, value: str) -> ParserError:
    """Return an instance of ParserError when parsing fails and no choices are available."""
    if self.no_match_message:
        return ParserError(self.no_match_message)
    return super().no_choices_available(value)