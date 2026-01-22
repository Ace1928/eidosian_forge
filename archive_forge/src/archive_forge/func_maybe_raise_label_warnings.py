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
def maybe_raise_label_warnings(label: str | None, label_visibility: str | None):
    if not label:
        _LOGGER.warning('`label` got an empty value. This is discouraged for accessibility reasons and may be disallowed in the future by raising an exception. Please provide a non-empty label and hide it with label_visibility if needed.')
    if label_visibility not in ('visible', 'hidden', 'collapsed'):
        raise errors.StreamlitAPIException(f"Unsupported label_visibility option '{label_visibility}'. Valid values are 'visible', 'hidden' or 'collapsed'.")