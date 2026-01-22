from __future__ import annotations
import argparse
from collections.abc import Iterable, Sequence
import sys
from markdown_it import __version__
from markdown_it.main import MarkdownIt
def interactive() -> None:
    """
    Parse user input, dump to stdout, rinse and repeat.
    Python REPL style.
    """
    print_heading()
    contents = []
    more = False
    while True:
        try:
            prompt, more = ('... ', True) if more else ('>>> ', True)
            contents.append(input(prompt) + '\n')
        except EOFError:
            print('\n' + MarkdownIt().render('\n'.join(contents)), end='')
            more = False
            contents = []
        except KeyboardInterrupt:
            print('\nExiting.')
            break