from __future__ import annotations
from io import IOBase
from lazyops.types import BaseModel, Field
from lazyops.utils import logger
from typing import Optional, Dict, Any, List, Union, Sequence, Callable, TYPE_CHECKING
from .types import SlackContext, SlackPayload
from .configs import SlackSettings
def temp_message(self, text: str, channel: str, attachments: Optional[Sequence[Union[Dict, 'Attachment']]]=None, blocks: Optional[Sequence[Union[Dict, 'Block']]]=None, **kwargs):
    """
        Send a temporary message
        """
    channel = self.client.lookup(channel)
    return self.client.sapi.chat_postEphemeral(text=text, channel=channel, user=self.uid, attachments=attachments, blocks=blocks, **kwargs)