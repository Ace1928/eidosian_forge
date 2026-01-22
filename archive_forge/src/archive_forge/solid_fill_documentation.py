from __future__ import annotations
from urwid.canvas import SolidCanvas
from .constants import SHADE_SYMBOLS, Sizing
from .widget import Widget

        Render the Fill as a canvas and return it.

        >>> SolidFill().render((4,2)).text # ... = b in Python 3
        [...'    ', ...'    ']
        >>> SolidFill('#').render((5,3)).text
        [...'#####', ...'#####', ...'#####']
        