from abc import ABC
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Type
def defaults_factory(args):
    return self.__class__(self.subparsers, args, command=self.COMMAND, from_defaults_factory=True, parser=self.parser)