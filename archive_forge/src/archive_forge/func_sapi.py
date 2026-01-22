from __future__ import annotations
from io import IOBase
from lazyops.types import BaseModel, Field
from lazyops.utils import logger
from typing import Optional, Dict, Any, List, Union, Sequence, Callable, TYPE_CHECKING
from .types import SlackContext, SlackPayload
from .configs import SlackSettings
@property
def sapi(self) -> 'WebClient':
    """
        Returns the Sync API
        """
    if self._sapi is None and (not self.disabled):
        from slack_sdk import WebClient
        self._sapi = WebClient(token=self.token, **self._kwargs)
    return self._sapi