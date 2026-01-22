import os
import subprocess
from IPython.core.error import TryNext
import IPython.utils.py3compat as py3compat
def win32_clipboard_get():
    """ Get the current clipboard's text on Windows.

    Requires Mark Hammond's pywin32 extensions.
    """
    try:
        import win32clipboard
    except ImportError as e:
        raise TryNext('Getting text from the clipboard requires the pywin32 extensions: http://sourceforge.net/projects/pywin32/') from e
    win32clipboard.OpenClipboard()
    try:
        text = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
    except (TypeError, win32clipboard.error):
        try:
            text = win32clipboard.GetClipboardData(win32clipboard.CF_TEXT)
            text = py3compat.cast_unicode(text, py3compat.DEFAULT_ENCODING)
        except (TypeError, win32clipboard.error) as e:
            raise ClipboardEmpty from e
    finally:
        win32clipboard.CloseClipboard()
    return text