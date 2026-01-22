from __future__ import annotations
import abc
import contextlib
import functools
import os
import platform
import selectors
import signal
import socket
import sys
import typing
from urwid import signals, str_util, util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, UPDATE_PALETTE_ENTRY, AttrSpec, BaseScreen, RealTerminal
def get_available_raw_input(self) -> list[int]:
    """
        Return any currently available input. Does not block.

        This method is only used by the default `hook_event_loop`
        implementation; you can safely ignore it if you implement your own.
        """
    logger = self.logger.getChild('get_available_raw_input')
    codes = [*self._partial_codes, *self._get_input_codes()]
    self._partial_codes = []
    with selectors.DefaultSelector() as selector:
        selector.register(self._resize_pipe_rd, selectors.EVENT_READ)
        present_resize_flag = selector.select(0)
        while present_resize_flag:
            logger.debug('Resize signal received. Cleaning socket.')
            self._resize_pipe_rd.recv(128)
            present_resize_flag = selector.select(0)
    return codes