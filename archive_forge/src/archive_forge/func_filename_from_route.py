from __future__ import annotations
import logging # isort:skip
import argparse
import sys
from abc import abstractmethod
from os.path import splitext
from ...document import Document
from ..subcommand import (
from ..util import build_single_handler_applications, die
def filename_from_route(self, route: str, ext: str) -> str:
    """

        """
    if route == '/':
        base = 'index'
    else:
        base = route[1:]
    return f'{base}.{ext}'