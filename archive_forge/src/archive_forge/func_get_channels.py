from __future__ import annotations
from io import IOBase
from lazyops.types import BaseModel, Field
from lazyops.utils import logger
from typing import Optional, Dict, Any, List, Union, Sequence, Callable, TYPE_CHECKING
from .types import SlackContext, SlackPayload
from .configs import SlackSettings
def get_channels(self, types: Optional[List[str]]=None, include_users: Optional[bool]=True, **kwargs) -> List[Dict[str, Any]]:
    """
        Get channels
        """
    if not types:
        types = ['public_channel', 'private_channel', 'im', 'mpim']
    if types and isinstance(types, list):
        types = ','.join(types)
    resp = self.sapi.conversations_list(types=types, **kwargs)
    channels = resp['channels'] if resp['ok'] else []
    if include_users:
        resp = self.sapi.users_conversations(types=types, **kwargs)
        channels += resp['channels'] if resp['ok'] else []
    return channels