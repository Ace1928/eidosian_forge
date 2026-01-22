import re
from git.cmd import handle_process_output
from git.compat import defenc
from git.util import finalize_process, hex_to_bin
from .objects.blob import Blob
from .objects.util import mode_str_to_int
from typing import (
from git.types import PathLike, Literal
@property
def b_path(self) -> Optional[str]:
    return self.b_rawpath.decode(defenc, 'replace') if self.b_rawpath else None