from __future__ import annotations
import argparse
from ...completers import (
from ...environments import (
from .network import (
from .posix import (
from .windows import (
def do_integration(subparsers, parent: argparse.ArgumentParser, completer: CompositeActionCompletionFinder):
    """Command line parsing for all integration commands."""
    parser = argparse.ArgumentParser(add_help=False, parents=[parent])
    do_posix_integration(subparsers, parser, add_integration_common, completer)
    do_network_integration(subparsers, parser, add_integration_common, completer)
    do_windows_integration(subparsers, parser, add_integration_common, completer)