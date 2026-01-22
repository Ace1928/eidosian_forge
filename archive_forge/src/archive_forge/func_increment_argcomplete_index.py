from __future__ import annotations
import argparse
import os
import typing as t
def increment_argcomplete_index() -> None:
    """Assumes ``$_ARGCOMPLETE`` is set and `argcomplete` is importable

    Increment the index pointed to by ``$_ARGCOMPLETE``, which is used to
    determine which word `argcomplete` should start evaluating the command-line.
    This may be useful to "inform" `argcomplete` that we have already evaluated
    the first word as a subcommand.
    """
    try:
        os.environ['_ARGCOMPLETE'] = str(int(os.environ['_ARGCOMPLETE']) + 1)
    except Exception:
        try:
            argcomplete.debug('Unable to increment $_ARGCOMPLETE', os.environ['_ARGCOMPLETE'])
        except (KeyError, ModuleNotFoundError):
            pass