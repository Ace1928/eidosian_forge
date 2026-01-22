from __future__ import annotations
import re
from typing import Any
from ..common.utils import charCodeAt, isMdAsciiPunct, isPunctChar, isWhiteSpace
from ..token import Token
from .state_core import StateCore
Convert straight quotation marks to typographic ones
