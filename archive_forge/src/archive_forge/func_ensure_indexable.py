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
def ensure_indexable(obj: OptionSequence[V_co]) -> Sequence[V_co]:
    """Try to ensure a value is an indexable Sequence. If the collection already
    is one, it has the index method that we need. Otherwise, convert it to a list.
    """
    it = ensure_iterable(obj)
    index_fn = getattr(it, 'index', None)
    if callable(index_fn):
        return copy.copy(cast(Sequence[V_co], it))
    else:
        return list(it)