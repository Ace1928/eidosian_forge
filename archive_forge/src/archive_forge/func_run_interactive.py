import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def run_interactive(self):
    """
        Starts interactive CLI mode.
        """
    history = self.prefs['path_history']
    index = self.prefs['path_history_index']
    if history and index:
        if index < len(history):
            try:
                target = self._root_node.get_node(history[index])
            except ValueError:
                self._current_node = self._root_node
            else:
                self._current_node = target
    while True:
        try:
            old_completer = readline.get_completer()
            self._cli_loop()
            break
        except KeyboardInterrupt:
            self.con.raw_write('\n')
        finally:
            readline.set_completer(old_completer)