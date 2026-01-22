from __future__ import annotations
import typing as t
from sqlglot import exp
from sqlglot.helper import tsort

    Remove INNER and OUTER from joins as they are optional.
    