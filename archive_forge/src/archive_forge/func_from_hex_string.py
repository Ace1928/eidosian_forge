from __future__ import annotations
import logging # isort:skip
import colorsys
from abc import ABCMeta, abstractmethod
from math import sqrt
from re import match
from typing import TYPE_CHECKING, Union
from ..core.serialization import AnyRep, Serializable, Serializer
from ..util.deprecation import deprecated
@classmethod
def from_hex_string(cls, hex_string: str) -> RGB:
    """ Create an RGB color from a RGB(A) hex string.

        Args:
            hex_string (str) :
                String containing hex-encoded RGBA(A) values. Valid formats
                are '#rrggbb', '#rrggbbaa', '#rgb' and '#rgba'.

        Returns:
            :class:`~bokeh.colors.RGB`

        """
    if isinstance(hex_string, str):
        if match('#([\\da-fA-F]{2}){3,4}\\Z', hex_string):
            r = int(hex_string[1:3], 16)
            g = int(hex_string[3:5], 16)
            b = int(hex_string[5:7], 16)
            a = int(hex_string[7:9], 16) / 255.0 if len(hex_string) > 7 else 1.0
            return RGB(r, g, b, a)
        if match('#[\\da-fA-F]{3,4}\\Z', hex_string):
            r = int(hex_string[1] * 2, 16)
            g = int(hex_string[2] * 2, 16)
            b = int(hex_string[3] * 2, 16)
            a = int(hex_string[4] * 2, 16) / 255.0 if len(hex_string) > 4 else 1.0
            return RGB(r, g, b, a)
    raise ValueError(f"'{hex_string}' is not an RGB(A) hex color string")