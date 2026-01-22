from __future__ import annotations
import io
import json
import logging
from typing import Any, Callable
import pytest
from jupyter_events import EventLogger
@pytest.fixture
def jp_event_logger(jp_event_handler: logging.Handler, jp_event_schemas: list[Any]) -> EventLogger:
    """A pre-configured event logger for tests."""
    logger = EventLogger()
    for schema in jp_event_schemas:
        logger.register_event_schema(schema)
    logger.register_handler(handler=jp_event_handler)
    return logger