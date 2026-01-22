import json
import re
import tempfile
from typing import Any, List, Optional
from click import echo, style
from mypy_extensions import mypyc_attr
@mypyc_attr(patchable=True)
def dump_to_file(*output: str, ensure_final_newline: bool=True) -> str:
    """Dump `output` to a temporary file. Return path to the file."""
    with tempfile.NamedTemporaryFile(mode='w', prefix='blk_', suffix='.log', delete=False, encoding='utf8') as f:
        for lines in output:
            f.write(lines)
            if ensure_final_newline and lines and (lines[-1] != '\n'):
                f.write('\n')
    return f.name