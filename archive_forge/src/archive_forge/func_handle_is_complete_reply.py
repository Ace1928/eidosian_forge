from __future__ import print_function
import asyncio
import base64
import errno
from getpass import getpass
from io import BytesIO
import os
from queue import Empty
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from warnings import warn
from typing import Dict as DictType, Any as AnyType
from zmq import ZMQError
from IPython.core import page
from traitlets import (
from traitlets.config import SingletonConfigurable
from .completer import ZMQCompleter
from .zmqhistory import ZMQHistoryManager
from . import __version__
from prompt_toolkit import __version__ as ptk_version
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER, EditingMode
from prompt_toolkit.filters import (
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts.prompt import PromptSession
from prompt_toolkit.shortcuts import print_formatted_text, CompleteStyle
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.layout.processors import (
from prompt_toolkit.styles import merge_styles
from prompt_toolkit.styles.pygments import (style_from_pygments_cls,
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.utils import suspend_to_background_supported
from pygments.styles import get_style_by_name
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from pygments.token import Token
from jupyter_console.utils import run_sync, ensure_async
def handle_is_complete_reply(self, msg_id, timeout=None):
    """
        Wait for a repsonse from the kernel, and return two values:
            more? - (boolean) should the frontend ask for more input
            indent - an indent string to prefix the input
        Overloaded methods may want to examine the comeplete source. Its is
        in the self._source_lines_buffered list.
        """
    msg = None
    try:
        kwargs = {'timeout': timeout}
        msg = run_sync(self.client.shell_channel.get_msg)(**kwargs)
    except Empty:
        warn('The kernel did not respond to an is_complete_request. Setting `use_kernel_is_complete` to False.')
        self.use_kernel_is_complete = False
        return (False, '')
    if msg['parent_header'].get('msg_id', None) != msg_id:
        warn('The kernel did not respond properly to an is_complete_request: %s.' % str(msg))
        return (False, '')
    else:
        status = msg['content'].get('status', None)
        indent = msg['content'].get('indent', '')
    if status == 'complete':
        return (False, indent)
    elif status == 'incomplete':
        return (True, indent)
    elif status == 'invalid':
        raise SyntaxError()
    elif status == 'unknown':
        return (False, indent)
    else:
        warn('The kernel sent an invalid is_complete_reply status: "%s".' % status)
        return (False, indent)