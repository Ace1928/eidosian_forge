import os
import subprocess
from IPython.core.error import TryNext
import IPython.utils.py3compat as py3compat
def wayland_clipboard_get():
    """Get the clipboard's text under Wayland using wl-paste command.

    This requires Wayland and wl-clipboard installed and running.
    """
    if os.environ.get('XDG_SESSION_TYPE') != 'wayland':
        raise TryNext('wayland is not detected')
    try:
        with subprocess.Popen(['wl-paste'], stdout=subprocess.PIPE) as p:
            raw, err = p.communicate()
            if p.wait():
                raise TryNext(err)
    except FileNotFoundError as e:
        raise TryNext('Getting text from the clipboard under Wayland requires the wl-clipboard extension: https://github.com/bugaevc/wl-clipboard') from e
    if not raw:
        raise ClipboardEmpty
    try:
        text = py3compat.decode(raw)
    except UnicodeDecodeError as e:
        raise ClipboardEmpty from e
    return text