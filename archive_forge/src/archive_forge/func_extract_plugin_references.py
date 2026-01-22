from __future__ import annotations
import collections
import collections.abc as c
import enum
import os
import re
import itertools
import abc
import typing as t
from .encoding import (
from .io import (
from .util import (
from .data import (
def extract_plugin_references(name: str, aliases: list[str]) -> list[tuple[str, str]]:
    """Return a list of plugin references found in the given integration test target name and aliases."""
    plugins = content_plugins()
    found: list[tuple[str, str]] = []
    for alias in [name] + aliases:
        plugin_type = 'modules'
        plugin_name = alias
        if plugin_name in plugins.get(plugin_type, {}):
            found.append((plugin_type, plugin_name))
        parts = alias.split('_')
        for type_length in (1, 2):
            if len(parts) > type_length:
                plugin_type = '_'.join(parts[:type_length])
                plugin_name = '_'.join(parts[type_length:])
                if plugin_name in plugins.get(plugin_type, {}):
                    found.append((plugin_type, plugin_name))
    return found