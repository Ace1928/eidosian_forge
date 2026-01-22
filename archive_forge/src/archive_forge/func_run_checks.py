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
def run_checks(self) -> None:
    """Run the actual checks with the FileChecker Manager.

        This method encapsulates the logic to make a
        :class:`~flake8.checker.Manger` instance run the checks it is
        managing.
        """
    assert self.file_checker_manager is not None
    self.file_checker_manager.start()
    try:
        self.file_checker_manager.run()
    except exceptions.PluginExecutionFailed as plugin_failed:
        print(str(plugin_failed))
        print('Run flake8 with greater verbosity to see more details')
        self.catastrophic_failure = True
    LOG.info('Finished running')
    self.file_checker_manager.stop()
    self.end_time = time.time()