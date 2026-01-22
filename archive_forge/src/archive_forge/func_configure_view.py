import sys
from . import core
from altair.utils import use_signature
from altair.utils.schemapi import Undefined, UndefinedType
from typing import Any, Sequence, List, Literal, Union
@use_signature(core.ViewConfig)
def configure_view(self, *args, **kwargs) -> Self:
    copy = self.copy(deep=['config'])
    if copy.config is Undefined:
        copy.config = core.Config()
    copy.config['view'] = core.ViewConfig(*args, **kwargs)
    return copy