from abc import ABC
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Type
@property
def registered_subcommands(self):
    if not hasattr(self, '_registered_subcommands'):
        self._registered_subcommands = []
    return self._registered_subcommands