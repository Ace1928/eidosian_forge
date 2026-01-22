import sys
from .colorama.win32 import windll
from .colorama.winterm import WinColor, WinStyle, WinTerm
def winset(reset=False, fore=None, back=None, style=None, stderr=False):
    if reset:
        winterm.reset_all()
    if fore is not None:
        winterm.fore(fore, stderr)
    if back is not None:
        winterm.back(back, stderr)
    if style is not None:
        winterm.style(style, stderr)