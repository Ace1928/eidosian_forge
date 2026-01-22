import re
from git.cmd import handle_process_output
from git.compat import defenc
from git.util import finalize_process, hex_to_bin
from .objects.blob import Blob
from .objects.util import mode_str_to_int
from typing import (
from git.types import PathLike, Literal
def decode_path(path: bytes, has_ab_prefix: bool=True) -> Optional[bytes]:
    if path == b'/dev/null':
        return None
    if path.startswith(b'"') and path.endswith(b'"'):
        path = path[1:-1].replace(b'\\n', b'\n').replace(b'\\t', b'\t').replace(b'\\"', b'"').replace(b'\\\\', b'\\')
    path = _octal_byte_re.sub(_octal_repl, path)
    if has_ab_prefix:
        assert path.startswith(b'a/') or path.startswith(b'b/')
        path = path[2:]
    return path