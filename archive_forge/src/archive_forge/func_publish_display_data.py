from __future__ import annotations
import logging # isort:skip
import json
import os
import urllib
from typing import (
from uuid import uuid4
from ..core.types import ID
from ..util.serialization import make_id
from ..util.warnings import warn
from .state import curstate
def publish_display_data(data: dict[str, Any], metadata: dict[Any, Any] | None=None, *, transient: dict[str, Any] | None=None, **kwargs: Any) -> None:
    """

    """
    from IPython.display import publish_display_data
    publish_display_data(data, metadata, transient=transient, **kwargs)