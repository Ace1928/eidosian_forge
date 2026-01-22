import asyncio
from typing import Awaitable, TypeVar, Optional, Callable
from google.api_core.exceptions import GoogleAPICallError, Unknown
from google.cloud.pubsublite.internal.wait_ignore_cancelled import wait_ignore_errors

        Run a polling loop, which runs poll_action forever unless this is failed.
        Args:
          poll_action: A callable returning an awaitable to run in a loop. Note that async functions which return once
          satisfy this.
        