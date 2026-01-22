from __future__ import annotations
import io
import json
import logging
from typing import Any, Callable
import pytest
from jupyter_events import EventLogger
@pytest.fixture
def jp_event_sink() -> io.StringIO:
    """A stream for capture events."""
    return io.StringIO()