from __future__ import annotations
import argparse
import enum
import functools
import logging
from typing import Any
from typing import Callable
from typing import Sequence
from flake8 import utils
from flake8.plugins.finder import Plugins
def register_plugins(self, plugins: Plugins) -> None:
    """Register the plugin options (if needed)."""
    groups: dict[str, argparse._ArgumentGroup] = {}

    def _set_group(name: str) -> None:
        try:
            self._current_group = groups[name]
        except KeyError:
            group = self.parser.add_argument_group(name)
            self._current_group = groups[name] = group
    for loaded in plugins.all_plugins():
        add_options = getattr(loaded.obj, 'add_options', None)
        if add_options:
            _set_group(loaded.plugin.package)
            add_options(self)
        if loaded.plugin.entry_point.group == 'flake8.extension':
            self.extend_default_select([loaded.entry_name])
    self._current_group = None