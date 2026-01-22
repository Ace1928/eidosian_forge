import sys
from . import core
from altair.utils import use_signature
from altair.utils.schemapi import Undefined, UndefinedType
from typing import Any, Sequence, List, Literal, Union
@use_signature(core.RectConfig)
def configure_arc(self, *args, **kwargs) -> Self:
    copy = self.copy(deep=['config'])
    if copy.config is Undefined:
        copy.config = core.Config()
    copy.config['arc'] = core.RectConfig(*args, **kwargs)
    return copy