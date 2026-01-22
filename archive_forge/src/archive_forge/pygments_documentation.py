from __future__ import annotations
from typing import TYPE_CHECKING
from .style import Style

    Turn e.g. `Token.Name.Exception` into `'pygments.name.exception'`.

    (Our Pygments lexer will also turn the tokens that pygments produces in a
    prompt_toolkit list of fragments that match these styling rules.)
    