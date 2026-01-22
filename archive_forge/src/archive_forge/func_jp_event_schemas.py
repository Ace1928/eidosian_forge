from __future__ import annotations
import io
import json
import logging
from typing import Any, Callable
import pytest
from jupyter_events import EventLogger
@pytest.fixture
def jp_event_schemas() -> list[Any]:
    """A list of schema references.

    Each item should be one of the following:
    - string of serialized JSON/YAML content representing a schema
    - a pathlib.Path object pointing to a schema file on disk
    - a dictionary with the schema data.
    """
    return []