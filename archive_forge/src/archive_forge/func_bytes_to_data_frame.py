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
def bytes_to_data_frame(source: bytes) -> DataFrame:
    """Convert bytes to pandas.DataFrame.

    Using this function in production needs to make sure that
    the pyarrow version >= 14.0.1.

    Parameters
    ----------
    source : bytes
        A bytes object to convert.

    """
    import pyarrow as pa
    reader = pa.RecordBatchStreamReader(source)
    return reader.read_pandas()