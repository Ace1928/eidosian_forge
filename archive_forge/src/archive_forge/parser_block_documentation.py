from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Callable
from . import rules_block
from .ruler import Ruler
from .rules_block.state_block import StateBlock
from .token import Token
from .utils import EnvType
Process input string and push block tokens into `outTokens`.