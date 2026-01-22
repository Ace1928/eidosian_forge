from __future__ import annotations
import os.path
import types
import zipimport
from typing import Iterable, TYPE_CHECKING
from coverage import env
from coverage.exceptions import CoverageException, NoSource
from coverage.files import canonical_filename, relative_filename, zip_location
from coverage.misc import expensive, isolate_module, join_regex
from coverage.parser import PythonParser
from coverage.phystokens import source_token_lines, source_encoding
from coverage.plugin import FileReporter
from coverage.types import TArc, TLineNo, TMorf, TSourceTokenLines
def get_python_source(filename: str) -> str:
    """Return the source code, as unicode."""
    base, ext = os.path.splitext(filename)
    if ext == '.py' and env.WINDOWS:
        exts = ['.py', '.pyw']
    else:
        exts = [ext]
    source_bytes: bytes | None
    for ext in exts:
        try_filename = base + ext
        if os.path.exists(try_filename):
            source_bytes = read_python_source(try_filename)
            break
        source_bytes = get_zip_bytes(try_filename)
        if source_bytes is not None:
            break
    else:
        raise NoSource(f"No source for code: '{filename}'.")
    source_bytes = source_bytes.replace(b'\x0c', b' ')
    source = source_bytes.decode(source_encoding(source_bytes), 'replace')
    if source and source[-1] != '\n':
        source += '\n'
    return source