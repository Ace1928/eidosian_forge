import inspect
import locale
import logging
import logging.handlers
import os
import sys
from cliff import _argparse
from . import complete
from . import help
from . import utils
def print_help_if_requested(self):
    """Print help and exits if deferred help is enabled and requested.

        '--help' shows the help message and exits:
         * without calling initialize_app if not self.deferred_help (default),
         * after initialize_app call if self.deferred_help,
         * during initialize_app call if self.deferred_help and subclass calls
           explicitly this method in initialize_app.
        """
    if self.deferred_help and self.options.deferred_help:
        action = help.HelpAction(None, None, default=self)
        action(self.parser, self.options, None, None)