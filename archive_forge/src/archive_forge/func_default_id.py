from __future__ import annotations
from io import IOBase
from lazyops.types import BaseModel, Field
from lazyops.utils import logger
from typing import Optional, Dict, Any, List, Union, Sequence, Callable, TYPE_CHECKING
from .types import SlackContext, SlackPayload
from .configs import SlackSettings
@property
def default_id(self) -> Optional[str]:
    """
        Default ID
        """
    if self.disabled:
        return
    return self.default_user_id or self.default_channel_id