import importlib.metadata
import platform
import sys
import warnings
from typing import Any, Dict
from .. import __version__, constants
def is_notebook() -> bool:
    """Return `True` if code is executed in a notebook (Jupyter, Colab, QTconsole).

    Taken from https://stackoverflow.com/a/39662359.
    Adapted to make it work with Google colab as well.
    """
    try:
        shell_class = get_ipython().__class__
        for parent_class in shell_class.__mro__:
            if parent_class.__name__ == 'ZMQInteractiveShell':
                return True
        return False
    except NameError:
        return False