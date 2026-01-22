from __future__ import annotations
import argparse
import json
import logging
import time
from typing import Sequence
import flake8
from flake8 import checker
from flake8 import defaults
from flake8 import exceptions
from flake8 import style_guide
from flake8.formatting.base import BaseFormatter
from flake8.main import debug
from flake8.options.parse_args import parse_args
from flake8.plugins import finder
from flake8.plugins import reporter
def report_errors(self) -> None:
    """Report all the errors found by flake8 3.0.

        This also updates the :attr:`result_count` attribute with the total
        number of errors, warnings, and other messages found.
        """
    LOG.info('Reporting errors')
    assert self.file_checker_manager is not None
    results = self.file_checker_manager.report()
    self.total_result_count, self.result_count = results
    LOG.info('Found a total of %d violations and reported %d', self.total_result_count, self.result_count)