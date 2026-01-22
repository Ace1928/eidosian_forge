import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
def use_color_for_parser(parser: argparse.ArgumentParser, color: bool) -> None:
    """Configure a parser whether to output in color from HelpFormatters."""
    setattr(parser, _color_attr, color)