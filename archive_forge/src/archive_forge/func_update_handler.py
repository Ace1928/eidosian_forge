import asyncio
import atexit
import concurrent.futures
import errno
import functools
import select
import socket
import sys
import threading
import typing
import warnings
from tornado.gen import convert_yielded
from tornado.ioloop import IOLoop, _Selectable
from typing import (
def update_handler(self, fd: Union[int, _Selectable], events: int) -> None:
    fd, fileobj = self.split_fd(fd)
    if events & IOLoop.READ:
        if fd not in self.readers:
            self.selector_loop.add_reader(fd, self._handle_events, fd, IOLoop.READ)
            self.readers.add(fd)
    elif fd in self.readers:
        self.selector_loop.remove_reader(fd)
        self.readers.remove(fd)
    if events & IOLoop.WRITE:
        if fd not in self.writers:
            self.selector_loop.add_writer(fd, self._handle_events, fd, IOLoop.WRITE)
            self.writers.add(fd)
    elif fd in self.writers:
        self.selector_loop.remove_writer(fd)
        self.writers.remove(fd)