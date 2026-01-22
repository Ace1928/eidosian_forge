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
def brightness(self) -> float:
    """ Perceived brightness of a color in [0, 1] range. """
    r, g, b = (self.r, self.g, self.b)
    return sqrt(0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2) / 255