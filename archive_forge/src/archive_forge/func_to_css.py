from __future__ import annotations
import logging # isort:skip
import colorsys
from abc import ABCMeta, abstractmethod
from math import sqrt
from re import match
from typing import TYPE_CHECKING, Union
from ..core.serialization import AnyRep, Serializable, Serializer
from ..util.deprecation import deprecated
def to_css(self) -> str:
    """ Generate the CSS representation of this HSL color.

        Returns:
            str, ``"hsl(...)"`` or ``"hsla(...)"``

        """
    if self.a == 1.0:
        return f'hsl({self.h}, {self.s * 100}%, {self.l * 100}%)'
    else:
        return f'hsla({self.h}, {self.s * 100}%, {self.l * 100}%, {self.a})'