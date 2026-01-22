from __future__ import annotations
import sys
from typing import (
import param
from .base import ReplacementPane

    The `Streamz` pane renders streamz `Stream` objects emitting arbitrary
    objects, unlike the DataFrame pane which specifically handles streamz
    DataFrame and Series objects and exposes various formatting objects.

    Reference: https://panel.holoviz.org/reference/panes/Streamz.html

    :Example:

    >>> Streamz(some_streamz_stream_object, always_watch=True)
    