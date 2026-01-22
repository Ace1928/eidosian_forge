from __future__ import annotations
import asyncio
from textual._xterm_parser import XTermParser
from textual.app import App
from textual.driver import Driver
from textual.events import Resize
from textual.geometry import Size
def disable_input(self):
    if self._input_watcher is None:
        return
    self._terminal.param.unwatch(self._input_watcher)
    self._input_watcher = None