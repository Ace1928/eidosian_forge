import os
import subprocess
from .errors import HookError
def prepare_msg(*args):
    import tempfile
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        f.write(args[0])
    return (path,)