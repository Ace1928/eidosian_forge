from __future__ import annotations
import heapq
import logging
import os
import sys
import time
import typing
import warnings
from contextlib import suppress
from urwid import display, signals
from urwid.command_map import Command, command_map
from urwid.display.common import INPUT_DESCRIPTORS_CHANGED
from urwid.util import StoppingContext, is_mouse_event
from urwid.widget import PopUpTarget
from .abstract_loop import ExitMainLoop
from .select_loop import SelectEventLoop
def entering_idle(self) -> None:
    """
        This method is called whenever the event loop is about to enter the
        idle state. :meth:`draw_screen` is called here to update the
        screen when anything has changed.
        """
    if self.screen.started:
        self.draw_screen()
    else:
        self.logger.debug(f'No redrawing screen: {self.screen!r} is not started.')