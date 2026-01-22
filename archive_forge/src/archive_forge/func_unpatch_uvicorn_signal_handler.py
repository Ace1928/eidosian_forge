import io
import logging
import re
from datetime import datetime, timezone
from functools import partial
from typing import (
import anyio
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool
from starlette.responses import Response
from starlette.types import Receive, Scope, Send
def unpatch_uvicorn_signal_handler():
    """restores original signal-handler and rolls back monkey-patching.
        Normally this should not be necessary.
        """
    Server.handle_exit = original_handler