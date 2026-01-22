from __future__ import annotations
import typing as t
from enum import auto
from sqlglot.helper import AutoName
def merge_errors(errors: t.Sequence[ParseError]) -> t.List[t.Dict[str, t.Any]]:
    return [e_dict for error in errors for e_dict in error.errors]