from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Any
@abc.abstractproperty
def shell_channel_class(self) -> type[ChannelABC]:
    pass