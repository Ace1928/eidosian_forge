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
def source_for_file(filename: str) -> str:
    """Return the source filename for `filename`.

    Given a file name being traced, return the best guess as to the source
    file to attribute it to.

    """
    if filename.endswith('.py'):
        return filename
    elif filename.endswith(('.pyc', '.pyo')):
        py_filename = filename[:-1]
        if os.path.exists(py_filename):
            return py_filename
        if env.WINDOWS:
            pyw_filename = py_filename + 'w'
            if os.path.exists(pyw_filename):
                return pyw_filename
        return py_filename
    return filename