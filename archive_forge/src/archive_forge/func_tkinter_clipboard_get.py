import os
import subprocess
from IPython.core.error import TryNext
import IPython.utils.py3compat as py3compat
def tkinter_clipboard_get():
    """ Get the clipboard's text using Tkinter.

    This is the default on systems that are not Windows or OS X. It may
    interfere with other UI toolkits and should be replaced with an
    implementation that uses that toolkit.
    """
    try:
        from tkinter import Tk, TclError
    except ImportError as e:
        raise TryNext('Getting text from the clipboard on this platform requires tkinter.') from e
    root = Tk()
    root.withdraw()
    try:
        text = root.clipboard_get()
    except TclError as e:
        raise ClipboardEmpty from e
    finally:
        root.destroy()
    text = py3compat.cast_unicode(text, py3compat.DEFAULT_ENCODING)
    return text