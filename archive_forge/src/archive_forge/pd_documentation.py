from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from .bases import Property
 Accept Pandas DataFrame values.

    This property only exists to support type validation, e.g. for "accepts"
    clauses. It is not serializable itself, and is not useful to add to
    Bokeh models directly.

    