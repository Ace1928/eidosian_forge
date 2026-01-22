import contextlib
import os
import sys
from typing import List, Optional, Type, Union
from . import i18n, option, osutils, trace
from .lazy_import import lazy_import
import breezy
from breezy import (
from . import errors, registry
from .hooks import Hooks
from .i18n import gettext
from .plugin import disable_plugins, load_plugins, plugin_name
def guess_command(cmd_name):
    """Guess what command a user typoed.

    Args:
      cmd_name: Command to search for
    Returns:
      None if no command was found, name of a command otherwise
    """
    names = set()
    for name in all_command_names():
        names.add(name)
        cmd = get_cmd_object(name)
        names.update(cmd.aliases)
    costs = {}
    import patiencediff
    for name in sorted(names):
        matcher = patiencediff.PatienceSequenceMatcher(None, cmd_name, name)
        distance = 0.0
        opcodes = matcher.get_opcodes()
        for opcode, l1, l2, r1, r2 in opcodes:
            if opcode == 'delete':
                distance += l2 - l1
            elif opcode == 'replace':
                distance += max(l2 - l1, r2 - l1)
            elif opcode == 'insert':
                distance += r2 - r1
            elif opcode == 'equal':
                distance -= 0.1 * (l2 - l1)
        costs[name] = distance
    costs.update(_GUESS_OVERRIDES.get(cmd_name, {}))
    costs = sorted(((costs[key], key) for key in costs))
    if not costs:
        return
    if costs[0][0] > 4:
        return
    candidate = costs[0][1]
    return candidate