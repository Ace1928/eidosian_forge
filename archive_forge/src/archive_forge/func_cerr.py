import sys
from .colorama.win32 import windll
from .colorama.winterm import WinColor, WinStyle, WinTerm
def cerr(*args):
    """Shorthand for cprint('stderr', ...)"""
    cprint('stderr', *args)