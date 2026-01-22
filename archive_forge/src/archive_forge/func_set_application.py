from __future__ import unicode_literals
import socket
import select
import threading
import os
import fcntl
from six import int2byte, text_type, binary_type
from codecs import getincrementaldecoder
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.eventloop.base import EventLoop
from prompt_toolkit.interface import CommandLineInterface, Application
from prompt_toolkit.layout.screen import Size
from prompt_toolkit.shortcuts import create_prompt_application
from prompt_toolkit.terminal.vt100_input import InputStream
from prompt_toolkit.terminal.vt100_output import Vt100_Output
from .log import logger
from .protocol import IAC, DO, LINEMODE, SB, MODE, SE, WILL, ECHO, NAWS, SUPPRESS_GO_AHEAD
from .protocol import TelnetProtocolParser
from .application import TelnetApplication
def set_application(self, app, callback=None):
    """
        Set ``CommandLineInterface`` instance for this connection.
        (This can be replaced any time.)

        :param cli: CommandLineInterface instance.
        :param callback: Callable that takes the result of the CLI.
        """
    assert isinstance(app, Application)
    assert callback is None or callable(callback)
    self.cli = CommandLineInterface(application=app, eventloop=self.eventloop, output=self.vt100_output)
    self.callback = callback
    cb = self.cli.create_eventloop_callbacks()
    inputstream = InputStream(cb.feed_key)
    stdin_decoder_cls = getincrementaldecoder(self.encoding)
    stdin_decoder = [stdin_decoder_cls()]
    self.cli._is_running = True

    def data_received(data):
        """ TelnetProtocolParser 'data_received' callback """
        assert isinstance(data, binary_type)
        try:
            result = stdin_decoder[0].decode(data)
            inputstream.feed(result)
        except UnicodeDecodeError:
            stdin_decoder[0] = stdin_decoder_cls()
            return ''

    def size_received(rows, columns):
        """ TelnetProtocolParser 'size_received' callback """
        self.size = Size(rows=rows, columns=columns)
        cb.terminal_size_changed()
    self.parser = TelnetProtocolParser(data_received, size_received)