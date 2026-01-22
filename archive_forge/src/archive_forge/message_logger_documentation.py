import logging
from typing import Any
from uvicorn._types import (
from uvicorn.logging import TRACE_LOG_LEVEL

    Return an ASGI message, with any body-type content omitted and replaced
    with a placeholder.
    