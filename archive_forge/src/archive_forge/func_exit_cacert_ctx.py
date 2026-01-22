import sys
import atexit
def exit_cacert_ctx() -> None:
    _CACERT_CTX.__exit__(None, None, None)