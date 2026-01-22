from __future__ import annotations
from collections import namedtuple
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from .._compat import DATACLASS_KWARGS
from ..common.utils import isMdAsciiPunct, isPunctChar, isWhiteSpace
from ..ruler import StateBase
from ..token import Token
from ..utils import EnvType

        Scan a sequence of emphasis-like markers, and determine whether
        it can start an emphasis sequence or end an emphasis sequence.

         - start - position to scan from (it should point at a valid marker);
         - canSplitWord - determine if these markers can be found inside a word

        