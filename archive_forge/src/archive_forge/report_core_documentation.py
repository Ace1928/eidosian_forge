from __future__ import annotations
import sys
from typing import (
from coverage.exceptions import NoDataError, NotPython
from coverage.files import prep_patterns, GlobMatcher
from coverage.misc import ensure_dir_for_file, file_be_gone
from coverage.plugin import FileReporter
from coverage.results import Analysis
from coverage.types import TMorf
Generate a report of `morfs`, written to `outfile`.