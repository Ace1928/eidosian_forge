import sys
from . import core
from altair.utils import use_signature
from altair.utils.schemapi import Undefined, UndefinedType
from typing import Any, Sequence, List, Literal, Union
@use_signature(core.AxisConfig)
def configure_axisYTemporal(self, *args, **kwargs) -> Self:
    copy = self.copy(deep=['config'])
    if copy.config is Undefined:
        copy.config = core.Config()
    copy.config['axisYTemporal'] = core.AxisConfig(*args, **kwargs)
    return copy