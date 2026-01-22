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
def is_graphviz_chart(obj: object) -> TypeGuard[graphviz.Graph | graphviz.Digraph]:
    """True if input looks like a GraphViz chart."""
    return is_type(obj, 'graphviz.dot.Graph') or is_type(obj, 'graphviz.dot.Digraph') or is_type(obj, 'graphviz.graphs.Graph') or is_type(obj, 'graphviz.graphs.Digraph')