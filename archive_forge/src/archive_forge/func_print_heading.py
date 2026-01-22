from __future__ import annotations
import argparse
from collections.abc import Iterable, Sequence
import sys
from markdown_it import __version__
from markdown_it.main import MarkdownIt
def print_heading() -> None:
    print('{} (interactive)'.format(version_str))
    print('Type Ctrl-D to complete input, or Ctrl-C to exit.')