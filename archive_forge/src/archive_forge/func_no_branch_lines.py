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
@expensive
def no_branch_lines(self) -> set[TLineNo]:
    assert self.coverage is not None
    no_branch = self.parser.lines_matching(join_regex(self.coverage.config.partial_list), join_regex(self.coverage.config.partial_always_list))
    return no_branch