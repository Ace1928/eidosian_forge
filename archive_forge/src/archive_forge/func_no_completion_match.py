from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
def no_completion_match(self, value: str) -> CompletionUnavailable:
    """Return an instance of CompletionUnavailable when no match was found for the given value."""
    if self.no_match_message:
        return CompletionUnavailable(message=self.no_match_message)
    return super().no_completion_match(value)