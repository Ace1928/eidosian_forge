from __future__ import annotations
import asyncio
from textual._xterm_parser import XTermParser
from textual.app import App
from textual.driver import Driver
from textual.events import Resize
from textual.geometry import Size
def stop_application_mode(self):
    self._terminal.param.unwatch(self._size_watcher)
    self._disable_bracketed_paste()
    self.disable_input()
    self.write('\x1b[?1049l' + '\x1b[?25h')
    self.flush()