from __future__ import annotations
import logging # isort:skip
import colorsys
from abc import ABCMeta, abstractmethod
from math import sqrt
from re import match
from typing import TYPE_CHECKING, Union
from ..core.serialization import AnyRep, Serializable, Serializer
from ..util.deprecation import deprecated
@property
def luminance(self) -> float:
    """ Perceived luminance of a color in [0, 1] range. """
    r, g, b = (self.r, self.g, self.b)
    return (0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2) / 255 ** 2.2