from __future__ import annotations
import contextlib
import copy
import math
import re
import types
from enum import Enum, EnumMeta, auto
from typing import (
from typing_extensions import TypeAlias, TypeGuard
import streamlit as st
from streamlit import config, errors
from streamlit import logger as _logger
from streamlit import string_util
from streamlit.errors import StreamlitAPIException
def check_python_comparable(seq: Sequence[Any]) -> None:
    """Check if the sequence elements support "python comparison".
    That means that the equality operator (==) returns a boolean value.
    Which is not True for e.g. numpy arrays and pandas series."""
    try:
        bool(seq[0] == seq[0])
    except LookupError:
        pass
    except ValueError:
        raise StreamlitAPIException(f'Invalid option type provided. Options must be comparable, returning a boolean when used with *==*. \n\nGot **{type(seq[0]).__name__}**, which cannot be compared. Refactor your code to use elements of comparable types as options, e.g. use indices instead.')