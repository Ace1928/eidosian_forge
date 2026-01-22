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
def handle_iopub(self, msg_id=''):
    """Process messages on the IOPub channel

           This method consumes and processes messages on the IOPub channel,
           such as stdout, stderr, execute_result and status.

           It only displays output that is caused by this session.
        """
    while run_sync(self.client.iopub_channel.msg_ready)():
        sub_msg = run_sync(self.client.iopub_channel.get_msg)()
        msg_type = sub_msg['header']['msg_type']
        if msg_type == 'execute_input':
            self.execution_count = int(sub_msg['content']['execution_count']) + 1
        if self.include_output(sub_msg):
            if msg_type == 'status':
                self._execution_state = sub_msg['content']['execution_state']
            elif msg_type == 'stream':
                if sub_msg['content']['name'] == 'stdout':
                    if self._pending_clearoutput:
                        print('\r', end='')
                        self._pending_clearoutput = False
                    print(sub_msg['content']['text'], end='')
                    sys.stdout.flush()
                elif sub_msg['content']['name'] == 'stderr':
                    if self._pending_clearoutput:
                        print('\r', file=sys.stderr, end='')
                        self._pending_clearoutput = False
                    print(sub_msg['content']['text'], file=sys.stderr, end='')
                    sys.stderr.flush()
            elif msg_type == 'execute_result':
                if self._pending_clearoutput:
                    print('\r', end='')
                    self._pending_clearoutput = False
                self.execution_count = int(sub_msg['content']['execution_count'])
                if not self.from_here(sub_msg):
                    sys.stdout.write(self.other_output_prefix)
                format_dict = sub_msg['content']['data']
                self.handle_rich_data(format_dict)
                if 'text/plain' not in format_dict:
                    continue
                sys.stdout.flush()
                sys.stderr.flush()
                self.print_out_prompt()
                text_repr = format_dict['text/plain']
                if '\n' in text_repr:
                    print()
                print(text_repr)
                if not self.from_here(sub_msg):
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    self.print_remote_prompt()
            elif msg_type == 'display_data':
                data = sub_msg['content']['data']
                handled = self.handle_rich_data(data)
                if not handled:
                    if not self.from_here(sub_msg):
                        sys.stdout.write(self.other_output_prefix)
                    if 'text/plain' in data:
                        print(data['text/plain'])
            elif msg_type == 'execute_input':
                content = sub_msg['content']
                ec = content.get('execution_count', self.execution_count - 1)
                sys.stdout.write('\n')
                sys.stdout.flush()
                self.print_remote_prompt(ec=ec)
                sys.stdout.write(content['code'] + '\n')
            elif msg_type == 'clear_output':
                if sub_msg['content']['wait']:
                    self._pending_clearoutput = True
                else:
                    print('\r', end='')
            elif msg_type == 'error':
                for frame in sub_msg['content']['traceback']:
                    print(frame, file=sys.stderr)