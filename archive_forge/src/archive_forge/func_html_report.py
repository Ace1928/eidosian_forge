from __future__ import annotations
import atexit
import collections
import contextlib
import os
import os.path
import platform
import signal
import sys
import threading
import time
import warnings
from types import FrameType
from typing import (
from coverage import env
from coverage.annotate import AnnotateReporter
from coverage.collector import Collector, HAS_CTRACER
from coverage.config import CoverageConfig, read_coverage_config
from coverage.context import should_start_context_test_function, combine_context_switchers
from coverage.data import CoverageData, combine_parallel_data
from coverage.debug import (
from coverage.disposition import disposition_debug_msg
from coverage.exceptions import ConfigError, CoverageException, CoverageWarning, PluginError
from coverage.files import PathAliases, abs_file, relative_filename, set_relative_directory
from coverage.html import HtmlReporter
from coverage.inorout import InOrOut
from coverage.jsonreport import JsonReporter
from coverage.lcovreport import LcovReporter
from coverage.misc import bool_or_none, join_regex
from coverage.misc import DefaultValue, ensure_dir_for_file, isolate_module
from coverage.multiproc import patch_multiprocessing
from coverage.plugin import FileReporter
from coverage.plugin_support import Plugins
from coverage.python import PythonFileReporter
from coverage.report import SummaryReporter
from coverage.report_core import render_report
from coverage.results import Analysis
from coverage.types import (
from coverage.xmlreport import XmlReporter
def html_report(self, morfs: Iterable[TMorf] | None=None, directory: str | None=None, ignore_errors: bool | None=None, omit: str | list[str] | None=None, include: str | list[str] | None=None, extra_css: str | None=None, title: str | None=None, skip_covered: bool | None=None, show_contexts: bool | None=None, contexts: list[str] | None=None, skip_empty: bool | None=None, precision: int | None=None) -> float:
    """Generate an HTML report.

        The HTML is written to `directory`.  The file "index.html" is the
        overview starting point, with links to more detailed pages for
        individual modules.

        `extra_css` is a path to a file of other CSS to apply on the page.
        It will be copied into the HTML directory.

        `title` is a text string (not HTML) to use as the title of the HTML
        report.

        See :meth:`report` for other arguments.

        Returns a float, the total percentage covered.

        .. note::

            The HTML report files are generated incrementally based on the
            source files and coverage results. If you modify the report files,
            the changes will not be considered.  You should be careful about
            changing the files in the report folder.

        """
    self._prepare_data_for_reporting()
    with override_config(self, ignore_errors=ignore_errors, report_omit=omit, report_include=include, html_dir=directory, extra_css=extra_css, html_title=title, html_skip_covered=skip_covered, show_contexts=show_contexts, report_contexts=contexts, html_skip_empty=skip_empty, precision=precision):
        reporter = HtmlReporter(self)
        ret = reporter.report(morfs)
        return ret