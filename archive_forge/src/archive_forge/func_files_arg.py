from __future__ import annotations
import logging # isort:skip
import argparse
import sys
from abc import abstractmethod
from os.path import splitext
from ...document import Document
from ..subcommand import (
from ..util import build_single_handler_applications, die
@classmethod
def files_arg(cls, output_type_name: str) -> Arg:
    """ Returns a positional arg for ``files`` to specify file inputs to
        the command.

        Subclasses should include this to their class ``args``.

        Example:

            .. code-block:: python

                class Foo(FileOutputSubcommand):

                    args = (

                        FileOutputSubcommand.files_arg("FOO"),

                        # more args for Foo

                    ) + FileOutputSubcommand.other_args()

        """
    return ('files', Argument(metavar='DIRECTORY-OR-SCRIPT', nargs='+', help='The app directories or scripts to generate %s for' % output_type_name, default=None))