import threading
import sys
from paste.util import filemixin
def uninstall_stdin():
    global _stdincatcher, _oldstdin, register_stdin, deregister_stdin
    if _stdincatcher:
        sys.stdin = _oldstdin
        _stdincatcher = _oldstdin = None
        register_stdin = deregister_stdin = not_installed_error_stdin