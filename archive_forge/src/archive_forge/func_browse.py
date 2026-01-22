import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def browse(port=0, *, open_browser=True, hostname='localhost'):
    """Start the enhanced pydoc web server and open a web browser.

    Use port '0' to start the server on an arbitrary port.
    Set open_browser to False to suppress opening a browser.
    """
    import webbrowser
    serverthread = _start_server(_url_handler, hostname, port)
    if serverthread.error:
        print(serverthread.error)
        return
    if serverthread.serving:
        server_help_msg = 'Server commands: [b]rowser, [q]uit'
        if open_browser:
            webbrowser.open(serverthread.url)
        try:
            print('Server ready at', serverthread.url)
            print(server_help_msg)
            while serverthread.serving:
                cmd = input('server> ')
                cmd = cmd.lower()
                if cmd == 'q':
                    break
                elif cmd == 'b':
                    webbrowser.open(serverthread.url)
                else:
                    print(server_help_msg)
        except (KeyboardInterrupt, EOFError):
            print()
        finally:
            if serverthread.serving:
                serverthread.stop()
                print('Server stopped')